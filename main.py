import torch
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import time

# Import our modules
from model_architecture import MultiTaskCascadedLeadFunnelModel, train_cascaded_model, cascade_rank_leads, \
    precision_at_k
from training_pipeline import (analyze_group_differences,
                               finetune_with_external_examples)
from data_preparation import (preprocess_data,
                              prepare_external_examples,
                              visualize_group_differences)


def parse_args():
    parser = argparse.ArgumentParser(description='Lead Funnel Prediction Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the lead dataset CSV')
    parser.add_argument('--dict_path', type=str, default='data_dictonary.csv', help='Path to the data dictionary CSV')
    parser.add_argument('--dict_map_path', type=str, default='data_dictionary_mapping.csv',
                        help='Path to the dictionary mapping CSV')
    parser.add_argument('--external_data_path', type=str, help='Path to external examples CSV for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--output_file', type=str, default='results.joblib', help='Filename for the single output file')

    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (faster on newer GPUs)')
    parser.add_argument('--gradient_accum', type=int, default=8,
                        help='Gradient accumulation steps (use higher values for larger batch sizes)')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Updated cascade parameters for 2k -> 1k -> 500 -> 250 flow
    parser.add_argument('--toured_k', type=int, default=1000, help='Number of leads to select at toured stage (from 2000)')
    parser.add_argument('--applied_k', type=int, default=500, help='Number of leads to select at applied stage (from 1000)')
    parser.add_argument('--rented_k', type=int, default=250, help='Number of leads to select at rented stage (from 500)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated list of seeds to run with')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    parser.add_argument('--target_toured', type=str, default='TOTAL_APPOINTMENT_COMPLETED',
                        help='Column name for toured target')
    parser.add_argument('--target_applied', type=str, default='TOTAL_APPLIED',
                        help='Column name for applied target')
    parser.add_argument('--target_rented', type=str, default='TOTAL_RENTED',
                        help='Column name for rented target')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode for dimension tracing')

    # NEW ARGS for multiple subsets
    parser.add_argument('--num_subsets', type=int, default=1,
                        help='How many subsets to create from the data')
    parser.add_argument('--subset_size', type=int, default=2000,
                        help='Initial size of each subset (default 2000 leads)')
    parser.add_argument('--balance_classes', type=bool, default=False,
                        help='Whether to balance classes in each subset')

    return parser.parse_args()


def calculate_metrics(dataloader, model, device, k_values=[10, 20, 50, 100]):
    """Calculate ROC-AUC, APR, and Precision@k metrics for all three stages"""
    model.eval()

    all_toured_preds = []
    all_applied_preds = []
    all_rented_preds = []
    all_toured_true = []
    all_applied_true = []
    all_rented_true = []

    with torch.no_grad():
        for batch in dataloader:
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
            toured_labels = batch[2].cpu().numpy()
            applied_labels = batch[3].cpu().numpy()
            rented_labels = batch[4].cpu().numpy()

            # Predict - support both standard and cascaded model interfaces
            outputs = model(cat_in, num_in)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                # Extract only predictions (first 3 outputs)
                toured_pred, applied_pred, rented_pred = outputs[:3]
            else:
                toured_pred, applied_pred, rented_pred = outputs

            all_toured_preds.append(toured_pred.cpu().numpy())
            all_toured_true.append(toured_labels)

            all_applied_preds.append(applied_pred.cpu().numpy())
            all_applied_true.append(applied_labels)

            all_rented_preds.append(rented_pred.cpu().numpy())
            all_rented_true.append(rented_labels)

    # Flatten predictions and labels
    all_toured_preds = np.vstack(all_toured_preds).flatten()
    all_toured_true = np.vstack(all_toured_true).flatten()
    all_applied_preds = np.vstack(all_applied_preds).flatten()
    all_applied_true = np.vstack(all_applied_true).flatten()
    all_rented_preds = np.vstack(all_rented_preds).flatten()
    all_rented_true = np.vstack(all_rented_true).flatten()

    # Calculate metrics
    metrics = {
        'toured': {
            'roc_auc': roc_auc_score(all_toured_true, all_toured_preds),
            'apr': average_precision_score(all_toured_true, all_toured_preds),
            'precision_at_k': {}
        },
        'applied': {
            'roc_auc': roc_auc_score(all_applied_true, all_applied_preds),
            'apr': average_precision_score(all_applied_true, all_applied_preds),
            'precision_at_k': {}
        },
        'rented': {
            'roc_auc': roc_auc_score(all_rented_true, all_rented_preds),
            'apr': average_precision_score(all_rented_true, all_rented_preds),
            'precision_at_k': {}
        }
    }

    # Calculate Precision@k for various k values
    for k in k_values:
        for stage, preds, true in [
            ('toured', all_toured_preds, all_toured_true),
            ('applied', all_applied_preds, all_applied_true),
            ('rented', all_rented_preds, all_rented_true)
        ]:
            if len(true) >= k:  # Only calculate if we have enough samples
                p_at_k = precision_at_k(true, preds, k)
                metrics[stage]['precision_at_k'][k] = p_at_k

    # Print metrics
    print(f"Metrics:")
    for stage in ['toured', 'applied', 'rented']:
        print(f"{stage.capitalize()} => ROC-AUC: {metrics[stage]['roc_auc']:.4f}, APR: {metrics[stage]['apr']:.4f}")
        if 'precision_at_k' in metrics[stage]:
            for k, value in metrics[stage]['precision_at_k'].items():
                print(f"  P@{k}: {value:.4f}", end=" ")
        print()

    return metrics


def enable_debug_mode(model):
    """Enable debug mode for the model to trace dimension shapes in forward()."""

    def debug_forward(self, categorical_inputs, numerical_inputs, lead_ids=None, is_training=True):
        print(f"\n--- Starting debug forward pass ---")
        print(f"Categorical inputs shape: {categorical_inputs.shape}")
        print(f"Numerical inputs shape: {numerical_inputs.shape}")

        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        print(f"After embedding layer shape: {x.shape}")

        x = self.projection(x)
        print(f"After projection shape: {x.shape}")

        # For multi-task model, track shared features
        if hasattr(self, 'shared_transformer'):
            shared_features = self.shared_transformer(x)
            print(f"After shared transformer shape: {shared_features.shape}")
            toured_features = self.transformer_toured(shared_features)
        else:
            toured_features = self.transformer_toured(x)

        print(f"After toured transformer shape: {toured_features.shape}")

        toured_pred = self.toured_head(toured_features)
        print(f"Toured pred shape: {toured_pred.shape}")

        # Continue with cascaded processing
        _, toured_indices = torch.topk(toured_pred.squeeze(), min(self.toured_k, categorical_inputs.size(0)))
        print(f"Selected {len(toured_indices)} leads after toured stage")

        # For multi-task model, use shared features
        if hasattr(self, 'shared_transformer'):
            applied_features = self.transformer_applied(shared_features[toured_indices])
        else:
            applied_features = self.transformer_applied(toured_features[toured_indices])

        print(f"After applied transformer shape: {applied_features.shape}")

        applied_pred_subset = self.applied_head(applied_features)
        print(f"Applied pred subset shape: {applied_pred_subset.shape}")

        _, applied_indices_local = torch.topk(applied_pred_subset.squeeze(),
                                              min(self.applied_k, len(toured_indices)))
        print(f"Selected {len(applied_indices_local)} leads after applied stage")

        applied_indices = toured_indices[applied_indices_local]

        # For multi-task model, use shared features
        if hasattr(self, 'shared_transformer'):
            rented_features = self.transformer_rented(shared_features[applied_indices])
        else:
            rented_features = self.transformer_rented(applied_features[applied_indices_local])

        print(f"After rented transformer shape: {rented_features.shape}")

        rented_pred_subset = self.rented_head(rented_features)
        print(f"Rented pred subset shape: {rented_pred_subset.shape}")

        print(f"--- Finished debug forward pass ---\n")

        # Create complete pred tensors
        device = categorical_inputs.device
        batch_size = categorical_inputs.size(0)

        applied_pred = torch.zeros((batch_size, 1), device=device)
        applied_pred[toured_indices] = applied_pred_subset

        rented_pred = torch.zeros((batch_size, 1), device=device)
        rented_pred[applied_indices] = rented_pred_subset

        return toured_pred, applied_pred, rented_pred, toured_indices, applied_indices, applied_indices_local

    import types
    model.original_forward = model.forward
    model.debug_forward = types.MethodType(debug_forward, model)
    model.forward = model.debug_forward
    return model


def finetune_cascaded_with_external_examples(model,
                                             external_dataset,
                                             optimizer,
                                             toured_weight=2.0,
                                             applied_weight=2.0,
                                             rented_weight=2.0,
                                             epochs=5,
                                             device='cuda'):
    """
    Modified version of finetune_with_external_examples that handles the cascaded model's
    different return signature
    """
    toured_criterion = torch.nn.BCELoss()
    applied_criterion = torch.nn.BCELoss()
    rented_criterion = torch.nn.BCELoss()

    # Adjusted to handle large datasets - use a more reasonable batch size for fine-tuning
    batch_size = min(256, len(external_dataset))
    ext_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=True)
    model.train()

    print(f"Fine-tuning with {len(external_dataset)} examples in batches of {batch_size}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in ext_loader:
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
            toured_labels = batch[2].to(device)
            applied_labels = batch[3].to(device)
            rented_labels = batch[4].to(device)
            lead_ids = batch[5].to(device) if len(batch) > 5 else None

            optimizer.zero_grad()
            # Handle the 6 outputs from cascaded model
            toured_pred, applied_pred, rented_pred, _, _, _ = model(
                cat_in, num_in, lead_ids, is_training=True
            )

            toured_loss = toured_weight * toured_criterion(toured_pred, toured_labels)
            applied_loss = applied_weight * applied_criterion(applied_pred, applied_labels)
            rented_loss = rented_weight * rented_criterion(rented_pred, rented_labels)

            # Extra scale factor for external examples
            loss = 3.0 * (toured_loss + applied_loss + rented_loss)

            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        print(f"Fine-tuning Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    return model


def train_and_evaluate(train_dataset, test_dataset, args, device, categorical_dims, numerical_dim, results,
                       subset_label="single"):
    """Train and evaluate a model for a single subset, storing results in the results dictionary"""

    # Create DataLoader with appropriate batch size
    # For very large datasets, adjust the batch size to be realistic for GPU memory
    effective_batch_size = min(args.batch_size, len(train_dataset))
    print(f"Using effective batch size of {effective_batch_size} (requested: {args.batch_size})")

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)

    # Build embedding dimensions
    embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]
    total_feature_dim = sum(embedding_dims) + numerical_dim

    # Ensure transformer_dim multiple of num_heads
    transformer_dim = 256
    num_heads = 4
    if transformer_dim % num_heads != 0:
        transformer_dim = (transformer_dim // num_heads) * num_heads

    # Initialize the multi-task cascaded model instead of regular cascaded model
    model = MultiTaskCascadedLeadFunnelModel(
        categorical_dims=categorical_dims,
        embedding_dims=embedding_dims,
        numerical_dim=numerical_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        ff_dim=512,
        head_hidden_dims=[128, 64],
        dropout=0.2,
        toured_k=args.toured_k,
        applied_k=args.applied_k,
        rented_k=args.rented_k
    ).to(device)

    if args.debug:
        model = enable_debug_mode(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Temporarily create model file for train_cascaded_model
    # We'll delete this file later after loading its contents
    temp_model_path = os.path.join(args.output_dir, f'temp_model_{subset_label}.pt')

    print(
        f"Training model with {len(train_dataset)} samples, toured_k={args.toured_k}, applied_k={args.applied_k}, rented_k={args.rented_k}")
    print(f"Using gradient accumulation steps: {args.gradient_accum}")

    # Use the cascaded training function with ranking loss
    model, history = train_cascaded_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        toured_weight=1.0,
        applied_weight=1.0,
        rented_weight=2.0,
        ranking_weight=0.3,  # Add 30% weight to ranking loss
        device=device,
        model_save_path=temp_model_path,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accum,
        verbose=True  # Enable verbose output
    )

    # Store the state dict bytes
    if os.path.exists(temp_model_path):
        with open(temp_model_path, 'rb') as f:
            model_state_bytes = f.read()
        # Remove the temp file
        os.remove(temp_model_path)
    else:
        # If the model file wasn't created for some reason, just use the current state
        model_state_bytes = None
        print(f"Warning: Temporary model file not found for {subset_label}")

    # If external data is given, do fine-tuning with the cascaded-compatible function
    if args.external_data_path:
        print(f"Fine-tuning with external data for subset={subset_label}...")
        external_data = pd.read_csv(args.external_data_path)
        external_dataset = prepare_external_examples(
            external_data,
            preprocessors_path=os.path.join(args.output_dir, 'preprocessors')
        )
        model = finetune_cascaded_with_external_examples(
            model,
            external_dataset,
            optimizer,
            toured_weight=2.0,
            applied_weight=2.0,
            rented_weight=2.0,
            epochs=5,
            device=device
        )

        # Store the fine-tuned state instead
        finetuned_state_buffer = torch.save(model.state_dict(), None)
        model_state_bytes = finetuned_state_buffer

    # Use the cascaded ranking function
    print(f"[Subset {subset_label}] Performing multi-stage ranking on the test set...")
    rankings = cascade_rank_leads(
        model,
        test_loader,
        device=device,
        silent=False
    )

    # Calculate and store metrics, including Precision@k
    print(f"[Subset {subset_label}] Calculating ROC-AUC, APR, and Precision@k metrics...")
    metrics = calculate_metrics(test_loader, model, device, k_values=[10, 20, 50, 100])

    # Store results in the results dictionary
    subset_results = {
        'history': history,
        'model_state_bytes': model_state_bytes,
        'rankings': rankings,
        'metrics': metrics
    }

    results[subset_label] = subset_results

    return model


def train_with_seed(args, seed=None):
    """Main training function that can be run with a specific seed"""

    if seed is not None:
        args.seed = seed

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    from cuda_check import check_cuda
    cuda_available = check_cuda()

    if args.device == 'cuda' and not cuda_available:
        print("CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"GPU Memory reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    # Dictionary to store all results
    results = {
        'metadata': {
            'args': vars(args),
            'seed': args.seed,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        }
    }

    # Preprocess data
    # This call can return either a single (train_ds, test_ds, cat_dims, num_dim, feat_names)
    # or a dict with multiple subsets if num_subsets>1
    preprocess_result = preprocess_data(
        data_path=args.data_path,
        dict_path=args.dict_path,
        dict_map_path=args.dict_map_path,
        target_cols=[args.target_toured, args.target_applied, args.target_rented],
        save_preprocessors=True,
        preprocessors_path=os.path.join(args.output_dir, 'preprocessors'),
        num_subsets=args.num_subsets,
        subset_size=args.subset_size,
        balance_classes=args.balance_classes,
        random_state=args.seed
    )

    # If we have a single subset
    if isinstance(preprocess_result, tuple):
        train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names = preprocess_result

        # Store preprocessing metadata
        results['preprocessing'] = {
            'categorical_dims': categorical_dims,
            'numerical_dim': numerical_dim,
            'feature_names': feature_names
        }

        # Train on the single dataset
        model = train_and_evaluate(
            train_dataset,
            test_dataset,
            args,
            device,
            categorical_dims,
            numerical_dim,
            results,
            subset_label="single"
        )

    else:
        # We have multiple subsets
        subsets_list = preprocess_result['subsets']
        categorical_dims = preprocess_result['categorical_dims']
        numerical_dim = preprocess_result['numerical_dim']
        feature_names = preprocess_result['feature_names']

        # Store preprocessing metadata
        results['preprocessing'] = {
            'categorical_dims': categorical_dims,
            'numerical_dim': numerical_dim,
            'feature_names': feature_names,
            'num_subsets': len(subsets_list)
        }

        # Train on each subset
        for i, (train_ds, test_ds) in enumerate(subsets_list, start=1):
            subset_label = f"subset{i}"
            print(f"\n=== Training on subset {i}/{len(subsets_list)}: {len(train_ds)} train leads ===")

            model = train_and_evaluate(
                train_ds,
                test_ds,
                args,
                device,
                categorical_dims,
                numerical_dim,
                results,
                subset_label=subset_label
            )

    # Save all results to a single joblib file
    result_path = os.path.join(args.output_dir, args.output_file)
    joblib.dump(results, result_path)
    print(f"All results saved to {result_path}")

    return results


def main():
    global args, device, categorical_dims, numerical_dim
    args = parse_args()

    # Handle multiple seeds if provided
    if args.seeds:
        # Handle both comma-separated and space-separated lists
        if ',' in args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
        else:
            seeds = [int(s.strip()) for s in args.seeds.split()]

        print(f"Running with multiple seeds: {seeds}")

        # Create a consolidated results dictionary
        all_seeds_results = {}

        for seed_idx, seed in enumerate(seeds):
            print(f"\n=== Running with seed {seed} ({seed_idx + 1}/{len(seeds)}) ===\n")

            # Use a modified output filename for this seed
            original_output_file = args.output_file
            args.output_file = f"seed_{seed}_" + original_output_file

            # Run training with this seed
            seed_results = train_with_seed(args, seed)
            all_seeds_results[f"seed_{seed}"] = seed_results

            # Restore original output filename
            args.output_file = original_output_file

        # Save consolidated results from all seeds
        consolidated_path = os.path.join(args.output_dir, f"consolidated_{args.output_file}")
        joblib.dump(all_seeds_results, consolidated_path)
        print(f"Consolidated results from all seeds saved to {consolidated_path}")
    else:
        # Single seed run
        train_with_seed(args)


if __name__ == "__main__":
    main()