import torch
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our modules
from model_architecture import LeadFunnelModel
from training_pipeline import train_model, rank_leads_multi_stage, analyze_group_differences, \
    finetune_with_external_examples
from data_preparation import preprocess_data, prepare_external_examples, visualize_group_differences


def parse_args():
    parser = argparse.ArgumentParser(description='Lead Funnel Prediction Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the lead dataset CSV')
    parser.add_argument('--dict_path', type=str, default='data_dictonary.csv', help='Path to the data dictionary CSV')
    parser.add_argument('--dict_map_path', type=str, default='data_dictionary_mapping.csv',
                        help='Path to the dictionary mapping CSV')
    parser.add_argument('--external_data_path', type=str, help='Path to external examples CSV for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (faster on newer GPUs)')
    parser.add_argument('--gradient_accum', type=int, default=1,
                        help='Gradient accumulation steps (use higher values for larger batch sizes)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--toured_k', type=int, default=500, help='Number of leads to select at toured stage')
    parser.add_argument('--applied_k', type=int, default=250, help='Number of leads to select at applied stage')
    parser.add_argument('--rented_k', type=int, default=125, help='Number of leads to select at rented stage')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--target_toured', type=str, default='TOTAL_APPOINTMENT_COMPLETED', help='Column name for toured target')
    parser.add_argument('--target_applied', type=str, default='TOTAL_APPLIED', help='Column name for applied target')
    parser.add_argument('--target_rented', type=str, default='TOTAL_RENTED', help='Column name for rented target')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for dimension tracing')
    return parser.parse_args()


def enable_debug_mode(model):
    """Enable debug mode for the model to trace dimensions"""

    def debug_forward(self, categorical_inputs, numerical_inputs):
        """Debug version of forward method with dimension printing"""
        print(f"\n--- Starting debug forward pass ---")
        print(f"Categorical inputs shape: {categorical_inputs.shape}")
        print(f"Numerical inputs shape: {numerical_inputs.shape}")

        # Embed features
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        print(f"After embedding layer shape: {x.shape}")

        # Project to transformer dimension
        x = self.projection(x)
        print(f"After projection shape: {x.shape}")

        # Apply transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            print(f"Before transformer block {i + 1} shape: {x.shape}")
            x = block(x)
            print(f"After transformer block {i + 1} shape: {x.shape}")

        # Apply prediction heads
        print(f"Before prediction heads shape: {x.shape}")
        toured_pred = self.toured_head(x)
        applied_pred = self.applied_head(x)
        rented_pred = self.rented_head(x)
        print(f"Toured predictions shape: {toured_pred.shape}")
        print(f"Applied predictions shape: {applied_pred.shape}")
        print(f"Rented predictions shape: {rented_pred.shape}")
        print(f"--- Finished debug forward pass ---\n")

        return toured_pred, applied_pred, rented_pred

    # Save the original forward method and replace with debug version
    model.original_forward = model.forward
    import types
    model.debug_forward = types.MethodType(debug_forward, model)
    model.forward = model.debug_forward

    return model


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Import the CUDA check utility
    from cuda_check import check_cuda

    # Check CUDA availability with detailed diagnostics
    cuda_available = check_cuda()

    # Choose device
    if args.device == 'cuda' and not cuda_available:
        print("CUDA requested but not available. Please check the diagnostics above.")
        print("Defaulting to CPU. This will be much slower!")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # If using CUDA, set some optimization flags
    if device.type == 'cuda':
        # Enable cuDNN auto-tuner to find the best algorithm
        torch.backends.cudnn.benchmark = True
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Print GPU memory info
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    # Preprocess the data
    train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names = preprocess_data(
        args.data_path,
        dict_path=args.dict_path,
        dict_map_path=args.dict_map_path,
        target_cols=[args.target_toured, args.target_applied, args.target_rented],
        save_preprocessors=True,
        preprocessors_path=os.path.join(args.output_dir, 'preprocessors')
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate embedding dimensions based on cardinality
    embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]

    # Calculate total feature dimension
    total_feature_dim = sum(embedding_dims) + numerical_dim

    # Initialize model - ensure transformer dimension is a multiple of num_heads
    transformer_dim = 256  # Make sure this is divisible by num_heads
    num_heads = 4  # heads = 4 means transformer_dim should be divisible by 4

    # Verify dimensions are compatible
    if transformer_dim % num_heads != 0:
        transformer_dim = (transformer_dim // num_heads) * num_heads
        print(f"Adjusted transformer dimension to {transformer_dim} to be divisible by {num_heads} heads")

    # Initialize model
    model = LeadFunnelModel(
        categorical_dims=categorical_dims,
        embedding_dims=embedding_dims,
        numerical_dim=numerical_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_transformer_blocks=3,
        ff_dim=512,
        head_hidden_dims=[128, 64],
        dropout=0.2
    ).to(device)

    # Enable debug mode if requested
    if args.debug:
        model = enable_debug_mode(model)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Train the model
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        toured_weight=1.0,
        applied_weight=1.0,
        rented_weight=2.0,  # Higher weight for rented prediction
        device=device,
        model_save_path=os.path.join(args.output_dir, 'best_model.pt'),
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accum
    )

    # If external examples are provided, use them for fine-tuning
    if args.external_data_path:
        print("Fine-tuning with external examples...")
        external_data = pd.read_csv(args.external_data_path)
        external_dataset = prepare_external_examples(
            external_data,
            preprocessors_path=os.path.join(args.output_dir, 'preprocessors')
        )

        # Fine-tune the model
        model = finetune_with_external_examples(
            model,
            external_dataset,
            optimizer,
            toured_weight=2.0,
            applied_weight=2.0,
            rented_weight=2.0,
            epochs=5,
            device=device
        )

        # Save fine-tuned model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'finetuned_model.pt'))

    # Perform multi-stage ranking
    print("Performing multi-stage ranking...")
    rankings = rank_leads_multi_stage(
        model,
        test_loader,
        toured_k=args.toured_k,
        applied_k=args.applied_k,
        rented_k=args.rented_k,
        device=device
    )

    # Save rankings
    for stage in ['toured', 'applied', 'rented']:
        pd.DataFrame({
            'lead_id': rankings[stage]['selected'],
            'score': rankings[stage]['scores'].flatten()
        }).to_csv(os.path.join(args.output_dir, f'{stage}_selected_leads.csv'), index=False)

    # Load the original data for analysis
    full_data = pd.read_csv(args.data_path)

    # Analyze and visualize differences between selected and excluded leads at each stage
    for stage in ['toured', 'applied', 'rented']:
        print(f"\nAnalyzing differences for {stage} stage...")
        top_diffs, all_diffs = analyze_group_differences(
            full_data,
            rankings[stage]['selected'],
            rankings[stage]['excluded'],
            feature_names,
            top_n=20
        )

        # Visualize differences
        visualize_group_differences(
            top_diffs,
            stage,
            save_path=os.path.join(args.output_dir, f'{stage}_differences.png')
        )

    # Plot learning curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['toured_auc'], label='Toured')
    plt.plot(history['applied_auc'], label='Applied')
    plt.plot(history['rented_auc'], label='Rented')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['toured_apr'], label='Toured')
    plt.plot(history['applied_apr'], label='Applied')
    plt.plot(history['rented_apr'], label='Rented')
    plt.title('Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'learning_curves.png'))

    # Print final performance summary
    print("\nFinal Performance Summary:")
    print(f"Toured - AUC: {history['toured_auc'][-1]:.4f}, APR: {history['toured_apr'][-1]:.4f}")
    print(f"Applied - AUC: {history['applied_auc'][-1]:.4f}, APR: {history['applied_apr'][-1]:.4f}")
    print(f"Rented - AUC: {history['rented_auc'][-1]:.4f}, APR: {history['rented_apr'][-1]:.4f}")

    print("\nModel training and evaluation complete!")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    # Example usage:
    # python main.py --data_path=person_hh_number_for_ml.csv --dict_path=data_dictonary.csv --dict_map_path=data_dictionary_mapping.csv --device=cuda
    main()