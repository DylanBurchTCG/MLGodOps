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
import csv

# Import our modules
from model_architecture import MultiTaskCascadedLeadFunnelModel, train_cascaded_model, cascade_rank_leads, \
    precision_at_k
from training_pipeline import (analyze_group_differences,
                               finetune_with_external_examples,
                               perform_stage_specific_clustering)
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

    # Funnel selection options - fixed count vs. percentage-based
    parser.add_argument('--use_percentages', action='store_true', 
                        help='Use percentage-based selection instead of fixed counts')
    parser.add_argument('--adapt_to_data', action='store_true',
                        help='Automatically adapt percentages to match actual data distribution')
    
    # Percentage-based funnel parameters
    parser.add_argument('--toured_pct', type=float, default=0.5, 
                        help='Percentage of leads to select for toured stage (default: 50%)')
    parser.add_argument('--applied_pct', type=float, default=0.5, 
                        help='Percentage of toured leads to select for applied stage (default: 50%)')
    parser.add_argument('--rented_pct', type=float, default=0.5, 
                        help='Percentage of applied leads to select for rented stage (default: 50%)')
    
    # Fixed count funnel parameters (traditional approach)
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

    parser.add_argument('--stage_analysis', action='store_true', help='Enable stage-specific analysis')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters for stage-specific analysis')
    parser.add_argument('--skip_umap', action='store_true',
                        help='Skip UMAP dimension reduction for clustering')
    
    # NEW ARGS for feature engineering
    parser.add_argument('--enhance_toured_features', action='store_true', default=True,
                       help='Enable enhanced feature engineering for toured stage')

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
            cat_in = batch[0].to(device).long()  # Ensure long type for embedding
            num_in = batch[1].to(device).float()  # Ensure float type for numerical
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

        # Ensure correct types
        categorical_inputs = categorical_inputs.long()
        numerical_inputs = numerical_inputs.float()

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
        # Determine the number of leads to select based on method (percentage or fixed count)
        if hasattr(self, 'use_percentages') and self.use_percentages:
            # Use percentage-based selection
            k_toured = max(1, int(categorical_inputs.size(0) * self.toured_pct))
            print(f"Using percentage-based selection: {self.toured_pct*100:.1f}% = {k_toured} leads")
        else:
            # Use fixed count selection
            k_toured = min(self.toured_k, categorical_inputs.size(0))
            print(f"Using fixed count selection: k={k_toured} leads")
            
        _, toured_indices = torch.topk(toured_pred.squeeze(), k_toured)
        print(f"Selected {len(toured_indices)} leads after toured stage")

        # For multi-task model, use shared features
        if hasattr(self, 'shared_transformer'):
            applied_features = self.transformer_applied(shared_features[toured_indices])
        else:
            applied_features = self.transformer_applied(toured_features[toured_indices])

        print(f"After applied transformer shape: {applied_features.shape}")

        applied_pred_subset = self.applied_head(applied_features)
        print(f"Applied pred subset shape: {applied_pred_subset.shape}")

        # Determine applied selection count
        if hasattr(self, 'use_percentages') and self.use_percentages:
            # Use percentage-based selection
            k_applied = max(1, int(len(toured_indices) * self.applied_pct))
            print(f"Using percentage-based selection: {self.applied_pct*100:.1f}% = {k_applied} leads")
        else:
            # Use fixed count selection
            k_applied = min(self.applied_k, len(toured_indices))
            print(f"Using fixed count selection: k={k_applied} leads")
            
        _, applied_indices_local = torch.topk(applied_pred_subset.squeeze(), k_applied)
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

        # Determine rented selection count
        if hasattr(self, 'use_percentages') and self.use_percentages:
            # Use percentage-based selection
            k_rented = max(1, int(len(applied_indices) * self.rented_pct))
            print(f"Would select {k_rented} leads for rented stage ({self.rented_pct*100:.1f}%)")
        else:
            # Use fixed count selection
            k_rented = min(self.rented_k, len(applied_indices))
            print(f"Would select {k_rented} leads for rented stage (fixed count)")

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
            cat_in = batch[0].to(device).long()  # Ensure long type for embedding
            num_in = batch[1].to(device).float()  # Ensure float type for numerical
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


def save_history_to_csv(history, csv_path):
    """Save training history to a CSV file"""
    with open(csv_path, 'w', newline='') as csvfile:
        # Create CSV headers
        fieldnames = ['epoch', 'train_loss', 'val_loss', 
                      'toured_auc', 'applied_auc', 'rented_auc',
                      'toured_apr', 'applied_apr', 'rented_apr',
                      'toured_p10', 'applied_p10', 'rented_p10',
                      'toured_p50', 'applied_p50', 'rented_p50',
                      'learning_rate']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each epoch's metrics
        num_epochs = len(history['train_loss'])
        for epoch in range(num_epochs):
            row = {
                'epoch': epoch + 1,
                'train_loss': history['train_loss'][epoch],
                'val_loss': history['val_loss'][epoch],
                'learning_rate': history['lr'][epoch] if 'lr' in history else 'N/A'
            }
            
            # Add metrics if they exist in history
            for metric in ['auc', 'apr', 'p10', 'p50']:
                for stage in ['toured', 'applied', 'rented']:
                    key = f'{stage}_{metric}'
                    row[key] = history[key][epoch] if key in history and epoch < len(history[key]) else 'N/A'
            
            writer.writerow(row)
    
    print(f"Training history saved to {csv_path}")


def save_metrics_to_csv(metrics, csv_path):
    """Save final metrics to a CSV file"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers
        writer.writerow(['Stage', 'Metric', 'Value'])
        
        # Write metrics for each stage
        for stage in ['toured', 'applied', 'rented']:
            writer.writerow([stage, 'ROC-AUC', metrics[stage]['roc_auc']])
            writer.writerow([stage, 'APR', metrics[stage]['apr']])
            
            # Write Precision@k values
            for k, value in metrics[stage]['precision_at_k'].items():
                writer.writerow([stage, f'Precision@{k}', value])
    
    print(f"Final metrics saved to {csv_path}")


def save_rankings_to_csv(rankings, csv_path):
    """Save ranking results to a CSV file"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers
        writer.writerow(['Stage', 'Selected Count', 'Total Count', 'Selection Rate', 'Avg Score'])
        
        # Calculate and write metrics for each stage
        stages = ['toured', 'applied', 'rented']
        
        # First calculate the total leads
        total_leads = len(rankings['toured']['selected']) + len(rankings['toured']['excluded'])
        
        for stage in stages:
            selected_count = len(rankings[stage]['selected'])
            
            # For toured, use the total. For others, use the previous stage selected count
            if stage == 'toured':
                base_count = total_leads
            elif stage == 'applied':
                base_count = len(rankings['toured']['selected'])
            else:  # rented
                base_count = len(rankings['applied']['selected'])
                
            selection_rate = (selected_count / base_count * 100) if base_count > 0 else 0
            avg_score = np.mean(rankings[stage]['scores']) if len(rankings[stage]['scores']) > 0 else 0
            
            writer.writerow([
                stage, 
                selected_count, 
                base_count, 
                f"{selection_rate:.2f}%", 
                f"{avg_score:.4f}"
            ])
    
    print(f"Ranking results saved to {csv_path}")


def save_consolidated_results(results, train_dataset, test_dataset, metrics, rankings, csv_path):
    """Save a consolidated summary of results to a CSV file"""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write general information
        writer.writerow(['Training Summary'])
        writer.writerow(['Train Dataset Size', len(train_dataset)])
        writer.writerow(['Test Dataset Size', len(test_dataset)])
        writer.writerow(['Timestamp', time.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        
        # Write stage metrics
        writer.writerow(['Stage Metrics'])
        writer.writerow(['Stage', 'ROC-AUC', 'APR', 'P@10', 'P@50', 'Selected', 'Selection Rate'])
        
        total_leads = len(rankings['toured']['selected']) + len(rankings['toured']['excluded'])
        
        for stage in ['toured', 'applied', 'rented']:
            # For toured, use the total. For others, use the previous stage selected count
            if stage == 'toured':
                base_count = total_leads
            elif stage == 'applied':
                base_count = len(rankings['toured']['selected'])
            else:  # rented
                base_count = len(rankings['applied']['selected'])
                
            selected_count = len(rankings[stage]['selected'])
            selection_rate = (selected_count / base_count * 100) if base_count > 0 else 0
            
            writer.writerow([
                stage.capitalize(),
                f"{metrics[stage]['roc_auc']:.4f}",
                f"{metrics[stage]['apr']:.4f}",
                f"{metrics[stage]['precision_at_k'][10] if 10 in metrics[stage]['precision_at_k'] else 'N/A'}",
                f"{metrics[stage]['precision_at_k'][50] if 50 in metrics[stage]['precision_at_k'] else 'N/A'}",
                selected_count,
                f"{selection_rate:.2f}%"
            ])
        
        # Add final conversion rate
        initial_leads = total_leads
        final_selected = len(rankings['rented']['selected'])
        total_conversion = (final_selected / initial_leads * 100) if initial_leads > 0 else 0
        writer.writerow(['Overall Conversion', f"{total_conversion:.2f}%"])
        
    print(f"Consolidated results saved to {csv_path}")


def calculate_actual_stage_rates(train_dataset, preprocessed_rates=None):
    """
    Calculate the actual positive rates for each stage in the dataset
    to help set more realistic selection thresholds.
    
    Args:
        train_dataset: The training dataset
        preprocessed_rates: Optional dictionary of rates from preprocessing
    """
    # If we have preprocessed rates, use them
    if preprocessed_rates:
        # Just make sure we have all the rates we need
        rates = preprocessed_rates.copy()
        
        # Add buffer values if not already present
        if 'toured_pct' not in rates:
            rates['toured_pct'] = min(1.0, rates.get('toured_rate', 0.1) * 1.2)  # 20% buffer
        
        if 'applied_pct' not in rates:
            # Use conditional rate if available
            if 'applied_given_toured' in rates and rates['applied_given_toured'] > 0:
                rates['applied_pct'] = min(1.0, rates['applied_given_toured'] * 1.2)
            else:
                rates['applied_pct'] = min(1.0, rates.get('applied_rate', 0.1) / max(0.001, rates.get('toured_rate', 0.1)) * 1.2)
        
        if 'rented_pct' not in rates:
            # Use conditional rate if available
            if 'rented_given_applied' in rates and rates['rented_given_applied'] > 0:
                rates['rented_pct'] = min(1.0, rates['rented_given_applied'] * 1.2)
            else:
                rates['rented_pct'] = min(1.0, rates.get('rented_rate', 0.1) / max(0.001, rates.get('applied_rate', 0.1)) * 1.2)
        
        return rates
    
    # Calculate from dataset if no preprocessed rates
    toured_labels = torch.stack([item[2] for item in train_dataset]).flatten()
    applied_labels = torch.stack([item[3] for item in train_dataset]).flatten()
    rented_labels = torch.stack([item[4] for item in train_dataset]).flatten()
    
    # Calculate positive rates
    toured_rate = toured_labels.float().mean().item()
    applied_rate = applied_labels.float().mean().item()
    rented_rate = rented_labels.float().mean().item()
    
    # Try to calculate conditional rates when possible
    # Get indices where toured=1 and applied=1
    toured_mask = (toured_labels == 1).flatten()
    applied_mask = (applied_labels == 1).flatten()
    
    # Calculate conditional rates
    applied_given_toured = applied_labels[toured_mask].float().mean().item() if toured_mask.sum() > 0 else 0
    rented_given_applied = rented_labels[applied_mask].float().mean().item() if applied_mask.sum() > 0 else 0
    
    rates = {
        'toured_rate': toured_rate,
        'applied_rate': applied_rate,
        'rented_rate': rented_rate,
        'applied_given_toured': applied_given_toured,
        'rented_given_applied': rented_given_applied,
        # Add buffer to ensure we don't miss candidates
        'toured_pct': min(1.0, toured_rate * 1.2),  # 20% buffer
        'applied_pct': min(1.0, applied_given_toured * 1.2) if applied_given_toured > 0 else min(1.0, applied_rate / max(0.001, toured_rate) * 1.2),
        'rented_pct': min(1.0, rented_given_applied * 1.2) if rented_given_applied > 0 else min(1.0, rented_rate / max(0.001, applied_rate) * 1.2)
    }
    
    return rates


def train_and_evaluate(train_dataset, test_dataset, args, device, categorical_dims, numerical_dim, results,
                       subset_label="single"):
    """Train and evaluate a model for a single subset, storing results in the results dictionary"""

    # Get preprocessed stage rates if available
    preprocessed_rates = results.get('preprocessing', {}).get('stage_rates', None)

    # NEW: Calculate actual positive rates for adaptive thresholds
    stage_rates = calculate_actual_stage_rates(train_dataset, preprocessed_rates)
    print(f"\nActual stage rates in training data:")
    print(f"  - Toured: {stage_rates['toured_rate']*100:.2f}% positive")
    print(f"  - Applied: {stage_rates['applied_rate']*100:.2f}% positive")
    print(f"  - Rented: {stage_rates['rented_rate']*100:.2f}% positive")
    
    if 'applied_given_toured' in stage_rates:
        print(f"  - Applied when toured: {stage_rates['applied_given_toured']*100:.2f}%")
    if 'rented_given_applied' in stage_rates:
        print(f"  - Rented when applied: {stage_rates['rented_given_applied']*100:.2f}%")
    
    # NEW: Option to override percentage thresholds with actual rates
    if args.use_percentages and getattr(args, 'adapt_to_data', False):
        args.toured_pct = stage_rates['toured_pct']
        args.applied_pct = stage_rates['applied_pct']
        args.rented_pct = stage_rates['rented_pct']
        print(f"\nAdapted selection percentages to data distribution:")
        print(f"  - Toured threshold: {args.toured_pct*100:.2f}%")
        print(f"  - Applied threshold: {args.applied_pct*100:.2f}%")
        print(f"  - Rented threshold: {args.rented_pct*100:.2f}%")

    # Create DataLoader with appropriate batch size
    # For very large datasets, adjust the batch size to be realistic for GPU memory
    effective_batch_size = min(args.batch_size, len(train_dataset))
    print(f"Using effective batch size of {effective_batch_size} (requested: {args.batch_size})")

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)

    # Build embedding dimensions
    embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]
    total_feature_dim = sum(embedding_dims) + numerical_dim

    # ENHANCED: Larger transformer dimensions for improved capacity
    transformer_dim = 320  # Increased from 256
    num_heads = 8  # Increased from 4
    if transformer_dim % num_heads != 0:
        transformer_dim = (transformer_dim // num_heads) * num_heads

    print("\n" + "="*80)
    print(f"TRAINING MODEL FOR {subset_label}")
    print("="*80)
    print(f"Dataset: {len(train_dataset)} training samples, {len(test_dataset)} testing samples")
    print(f"Features: {len(categorical_dims)} categorical, {numerical_dim} numerical")
    print(f"Using device: {device}, Mixed precision: {args.mixed_precision}")
    print(f"Training config: batch_size={effective_batch_size}, grad_accum={args.gradient_accum}")
    
    # Print the selection approach being used
    if args.use_percentages:
        print(f"Selection by percentages: {args.toured_pct*100:.1f}% -> {args.applied_pct*100:.1f}% -> {args.rented_pct*100:.1f}%")
    else:
        print(f"Selection by fixed counts: {args.toured_k} -> {args.applied_k} -> {args.rented_k}")
    
    print("="*80 + "\n")
    
    # Initialize the multi-task cascaded model with updated parameters
    model = MultiTaskCascadedLeadFunnelModel(
        categorical_dims=categorical_dims,
        embedding_dims=embedding_dims,
        numerical_dim=numerical_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        ff_dim=512,
        head_hidden_dims=[128, 64],
        dropout=0.3,  # Increased dropout for better regularization
        toured_k=args.toured_k,
        applied_k=args.applied_k,
        rented_k=args.rented_k,
        use_percentages=args.use_percentages,  # Pass new percentage flag
        toured_pct=args.toured_pct,            # Pass percentage values
        applied_pct=args.applied_pct,
        rented_pct=args.rented_pct
    ).to(device)

    # Print model size
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {model_size:,} parameters")

    if args.debug:
        model = enable_debug_mode(model)

    # ENHANCED: Improved optimizer with weight decay and gradient clipping
    # Use a lower learning rate and clip gradients more aggressively
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01,  # Add weight decay for regularization
        eps=1e-8  # For numeric stability
    )
    
    # Use a more robust scheduler with patience
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=4,  # Increased from 3
        verbose=True,
        min_lr=1e-6
    )

    # Temporarily create model file for train_cascaded_model
    # We'll delete this file later after loading its contents
    temp_model_path = os.path.join(args.output_dir, f'temp_model_{subset_label}.pt')

    # Prepare CSV file paths
    history_csv_path = os.path.join(args.output_dir, f'{subset_label}_history.csv')
    metrics_csv_path = os.path.join(args.output_dir, f'{subset_label}_metrics.csv')
    rankings_csv_path = os.path.join(args.output_dir, f'{subset_label}_rankings.csv')
    summary_csv_path = os.path.join(args.output_dir, f'{subset_label}_summary.csv')
    epoch_csv_path = os.path.join(args.output_dir, f'{subset_label}_epochs.csv')

    print(f"Starting training with {len(train_dataset)} samples...")
    print(f"Training configuration:")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {effective_batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accum}")
    print(f"  - Mixed precision: {args.mixed_precision}")
    
    if args.use_percentages:
        print(f"  - Selection percentages: {args.toured_pct*100:.1f}% -> {args.applied_pct*100:.1f}% -> {args.rented_pct*100:.1f}%")
    else:
        print(f"  - Selection stages: {args.toured_k} -> {args.applied_k} -> {args.rented_k}")
    
    print(f"  - CSV output will be saved to {args.output_dir}\n")

    # Use the cascaded training function with ranking loss
    # ENHANCED: Increased weight for toured stage to 3.0 (from 1.0)
    model, history = train_cascaded_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        toured_weight=3.0,  # Increased weight for toured stage
        applied_weight=1.0,
        rented_weight=2.0,
        ranking_weight=0.4,  # Increased from 0.3 to emphasize ranking performance
        device=device,
        model_save_path=temp_model_path,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accum,
        verbose=True,  # Always enable verbose output
        epoch_csv_path=epoch_csv_path  # Add epoch-by-epoch CSV logging
    )

    # Save training history to CSV
    save_history_to_csv(history, history_csv_path)

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
            toured_weight=3.0,  # Match the higher toured weight
            applied_weight=1.0,
            rented_weight=2.0,
            epochs=5,
            device=device
        )

        # Store the fine-tuned state instead
        finetuned_state_buffer = torch.save(model.state_dict(), None)
        model_state_bytes = finetuned_state_buffer

    # Use the cascaded ranking function
    print(f"\n{'-'*80}")
    print(f"[Subset {subset_label}] Performing multi-stage ranking on the test set...")
    rankings = cascade_rank_leads(
        model,
        test_loader,
        device=device,
        silent=False
    )

    # Save rankings to CSV
    save_rankings_to_csv(rankings, rankings_csv_path)

    # Calculate and store metrics, including Precision@k
    print(f"\n{'-'*80}")
    print(f"[Subset {subset_label}] Calculating ROC-AUC, APR, and Precision@k metrics...")
    metrics = calculate_metrics(test_loader, model, device, k_values=[10, 20, 50, 100])

    # Save metrics to CSV
    save_metrics_to_csv(metrics, metrics_csv_path)

    # Create a consolidated summary CSV
    save_consolidated_results(results, train_dataset, test_dataset, metrics, rankings, summary_csv_path)
    
    # Store the initial results
    subset_results = {
        'history': history,
        'model_state_bytes': model_state_bytes,
        'rankings': rankings,
        'metrics': metrics,
        'csv_files': {
            'history': history_csv_path,
            'metrics': metrics_csv_path,
            'rankings': rankings_csv_path,
            'summary': summary_csv_path
        }
    }
    
    # NEW: Perform stage-specific analysis and cluster evaluation
    if getattr(args, 'stage_analysis', False):
        try:
            # Get feature names for analysis
            feature_names = results.get('preprocessing', {}).get('feature_names', [])
            
            print("\nPerforming stage-specific cluster analysis...")
            stage_analysis_results = perform_stage_specific_analysis(
                model,
                train_dataset,
                test_dataset,
                args,
                device,
                feature_names,
                subset_label=subset_label
            )
            
            # Create a mapping from lead_ids to clusters for each stage
            print("\nEvaluating cluster-specific performance metrics...")
            cluster_mappings = {}
            for stage in ['toured', 'applied', 'rented']:
                if stage in stage_analysis_results:
                    stage_data = stage_analysis_results[stage]
                    # Map each lead ID to its cluster
                    cluster_mappings[stage] = {
                        lead_id: cluster 
                        for lead_id, cluster in zip(stage_data['lead_ids'], stage_data['cluster_labels'])
                    }
                    
                    # Evaluate metrics for each cluster
                    if len(cluster_mappings[stage]) > 0:
                        cluster_metrics = evaluate_cluster_specific_metrics(
                            model, 
                            test_loader,
                            cluster_mappings[stage],
                            device=device
                        )
                        # Store cluster-specific metrics
                        stage_analysis_results[stage]['cluster_metrics'] = cluster_metrics
            
            # Store the analysis results
            subset_results['stage_analysis'] = stage_analysis_results
            
            # Save stage analysis to a separate file for easier access
            stage_analysis_path = os.path.join(args.output_dir, f'{subset_label}_stage_analysis.joblib')
            joblib.dump(stage_analysis_results, stage_analysis_path)
            print(f"Saved detailed stage analysis to {stage_analysis_path}")
            
        except Exception as e:
            print(f"Warning: Stage analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing without complete stage analysis...")
    
    # Store results in the results dictionary
    results[subset_label] = subset_results

    print(f"\nResult CSV files created:")
    print(f"  - Training history: {history_csv_path}")
    print(f"  - Metrics: {metrics_csv_path}")
    print(f"  - Rankings: {rankings_csv_path}")
    print(f"  - Summary: {summary_csv_path}")
    
    # Print final metrics
    print("\nFinal metrics summary:")
    print("="*40)
    print(f"Toured stage:  AUC={metrics['toured']['roc_auc']:.4f}  APR={metrics['toured']['apr']:.4f}")
    print(f"Applied stage: AUC={metrics['applied']['roc_auc']:.4f}  APR={metrics['applied']['apr']:.4f}")
    print(f"Rented stage:  AUC={metrics['rented']['roc_auc']:.4f}  APR={metrics['rented']['apr']:.4f}")
    print("="*40)

    return model


def train_with_seed(args, seed=None):
    """Main training function that can be run with a specific seed"""

    if seed is not None:
        args.seed = seed

    # Print header for this training run
    print("\n" + "="*80)
    print(f"STARTING TRAINING RUN WITH SEED {args.seed}")
    print("="*80 + "\n")

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

    # Print key configuration parameters
    print(f"\nConfiguration:")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Output file: {args.output_file}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accum}")
    
    # Only show the selection method being used
    if args.use_percentages:
        print(f"  - Using percentage-based selection")
        if args.adapt_to_data:
            print(f"  - Adapting percentages to data distribution")
        else:
            print(f"  - Percentages: {args.toured_pct*100:.1f}% -> {args.applied_pct*100:.1f}% -> {args.rented_pct*100:.1f}%")
    else:
        print(f"  - Using fixed count selection: {args.toured_k} -> {args.applied_k} -> {args.rented_k}")
        
    print(f"  - Enhanced toured features: {args.enhance_toured_features}")

    # Preprocess data
    # This call can return either a single (train_ds, test_ds, cat_dims, num_dim, feat_names)
    # or a dict with multiple subsets if num_subsets>1
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80 + "\n")
    
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
        enhance_toured_features=args.enhance_toured_features,
        # Add percentage-based selection parameters
        use_percentages=args.use_percentages,
        toured_pct=args.toured_pct,
        applied_pct=args.applied_pct,
        rented_pct=args.rented_pct,
        # Add fixed-count parameters too
        toured_k=args.toured_k,
        applied_k=args.applied_k,
        rented_k=args.rented_k,
        random_state=args.seed
    )

    # If we have a single subset
    if isinstance(preprocess_result, tuple):
        train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names, stage_rates = preprocess_result

        # Store preprocessing metadata
        results['preprocessing'] = {
            'categorical_dims': categorical_dims,
            'numerical_dim': numerical_dim,
            'feature_names': feature_names,
            'stage_rates': stage_rates
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
        stage_rates = preprocess_result.get('stage_rates', {})

        # Store preprocessing metadata
        results['preprocessing'] = {
            'categorical_dims': categorical_dims,
            'numerical_dim': numerical_dim,
            'feature_names': feature_names,
            'num_subsets': len(subsets_list),
            'stage_rates': stage_rates
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


def create_seed_summary(all_seeds_results, output_path):
    """Create a summary CSV file with the key metrics from all seed runs"""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Seed', 'Stage', 'ROC-AUC', 'APR', 'P@50', 'Final Train Loss', 'Final Val Loss'])
        
        # Write data for each seed
        for seed_key, seed_results in all_seeds_results.items():
            seed = seed_key.replace('seed_', '')
            
            # Find the subset results (single or subset1 if multiple)
            subset_key = 'single' if 'single' in seed_results else 'subset1'
            if subset_key in seed_results:
                subset_results = seed_results[subset_key]
                metrics = subset_results['metrics']
                history = subset_results['history']
                
                # Get final loss values
                final_train_loss = history['train_loss'][-1] if 'train_loss' in history and history['train_loss'] else 'N/A'
                final_val_loss = history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else 'N/A'
                
                # Add rows for each stage
                for stage in ['toured', 'applied', 'rented']:
                    if stage in metrics:
                        stage_metrics = metrics[stage]
                        auc = stage_metrics['roc_auc']
                        apr = stage_metrics['apr']
                        p50 = stage_metrics['precision_at_k'].get(50, 'N/A')
                        
                        writer.writerow([
                            seed,
                            stage,
                            f"{auc:.4f}",
                            f"{apr:.4f}",
                            f"{p50:.4f}" if p50 != 'N/A' else 'N/A',
                            f"{final_train_loss:.4f}" if final_train_loss != 'N/A' else 'N/A',
                            f"{final_val_loss:.4f}" if final_val_loss != 'N/A' else 'N/A'
                        ])
    
    print(f"Summary of all seed runs saved to {output_path}")


def perform_stage_specific_analysis(model, train_dataset, test_dataset, args, device, feature_names, subset_label="single"):
    """
    Perform stage-specific analysis including clustering for better interpretability.
    """
    print("\n" + "="*80)
    print(f"PERFORMING STAGE-SPECIFIC ANALYSIS FOR {subset_label}")
    print("="*80)
    
    # Create separate dataloaders for clarity
    train_loader = DataLoader(train_dataset, batch_size=min(1024, len(train_dataset)), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(1024, len(test_dataset)), shuffle=False)
    
    # Analyze each stage
    stages = ['toured', 'applied', 'rented']
    n_clusters = args.n_clusters if hasattr(args, 'n_clusters') else 8
    use_umap = not getattr(args, 'skip_umap', False)  # Use UMAP unless explicitly skipped
    
    results = {}
    
    # Run clustering on each stage using test set
    for stage in stages:
        print(f"\nAnalyzing {stage.upper()} stage:")
        cluster_results = perform_stage_specific_clustering(
            test_loader, 
            model, 
            device,
            n_clusters=n_clusters,
            stage=stage,
            use_umap=use_umap,
            random_state=args.seed
        )
        
        # Save results
        results[stage] = cluster_results
        
        # If we have plot bytes, save them as an image
        if cluster_results['plot_bytes']:
            cluster_dir = os.path.join(args.output_dir, 'clusters')
            os.makedirs(cluster_dir, exist_ok=True)
            plot_path = os.path.join(cluster_dir, f"{subset_label}_{stage}_clusters.png")
            
            with open(plot_path, 'wb') as f:
                f.write(cluster_results['plot_bytes'])
            print(f"Saved cluster visualization to {plot_path}")
        
        # Print some stats about the clusters
        print(f"Created {cluster_results['n_clusters']} clusters for {stage} stage")
        print("Cluster sizes:")
        for i, count in cluster_results['cluster_counts'].items():
            print(f"  - Cluster {i}: {count} leads")
    
    # Save all results
    analysis_path = os.path.join(args.output_dir, f"{subset_label}_stage_analysis.joblib")
    joblib.dump(results, analysis_path)
    print(f"\nSaved stage analysis results to {analysis_path}")
    
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
        
        # Create a summary CSV report
        seed_summary_path = os.path.join(args.output_dir, "seed_summary.csv")
        create_seed_summary(all_seeds_results, seed_summary_path)
    else:
        # Single seed run
        train_with_seed(args)


if __name__ == "__main__":
    main()