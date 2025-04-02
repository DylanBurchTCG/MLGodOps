import argparse
import torch
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tour_data_preparation import prepare_tour_data
from prospect_to_tour_model import TourPredictionModel
from tour_prediction_training import train_tour_prediction_model, TourDataset, precision_at_k

def parse_args():
    parser = argparse.ArgumentParser(description='Prospect to Tour Prediction Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the lead dataset CSV')
    parser.add_argument('--dict_path', type=str, default='data_dictonary.csv', help='Path to the data dictionary CSV')
    parser.add_argument('--dict_map_path', type=str, default='data_dictionary_mapping.csv',
                        help='Path to the dictionary mapping CSV')
    parser.add_argument('--output_dir', type=str, default='./tour_model_output', help='Directory to save model and results')
    parser.add_argument('--output_file', type=str, default='results.joblib', help='Name of the output file to save results')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated list of random seeds for multiple runs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--target_toured', type=str, default='TOTAL_APPOINTMENT_COMPLETED',
                        help='Column name for toured target')
    parser.add_argument('--enhance_features', action='store_true', default=True,
                        help='Enable enhanced feature engineering for toured stage')
    parser.add_argument('--use_percentages', action='store_true', default=False,
                        help='Use percentage-based metrics for evaluation')
    parser.add_argument('--adapt_to_data', action='store_true', default=False,
                        help='Adapt model parameters based on data characteristics')
    parser.add_argument('--model_name', type=str, default='tour_prediction_model.pt',
                        help='Name for saving the model')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss (class balance)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss (easy vs hard examples)')
    parser.add_argument('--hidden_layers', type=str, default='512,256,128',
                        help='Comma-separated list of hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse seeds
    seeds = [int(seed) for seed in args.seeds.split(',')]
    
    # Store results for all seeds
    all_results = []
    
    for seed in seeds:
        print(f"\nRunning with seed: {seed}")
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Use CPU if CUDA not available
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("\n" + "="*80)
        print(f"PROSPECT TO TOUR PREDICTION MODEL (Seed: {seed})")
        print("="*80)
        
        print(f"\nLoading and preprocessing data from {args.data_path}...")
        
        # Preprocess data with focus only on toured prediction
        train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names = prepare_tour_data(
            data_path=args.data_path,
            dict_path=args.dict_path,
            dict_map_path=args.dict_map_path,
            target_col=args.target_toured,
            enhance_features=args.enhance_features,
            random_state=seed
        )
        
        # Modify datasets to include only toured labels
        train_tour_dataset = TourDataset(
            categorical_features=torch.stack([item[0] for item in train_dataset]),
            numerical_features=torch.stack([item[1] for item in train_dataset]),
            toured_labels=torch.stack([item[2] for item in train_dataset]),
            lead_ids=torch.stack([item[5] for item in train_dataset])
        )
        
        test_tour_dataset = TourDataset(
            categorical_features=torch.stack([item[0] for item in test_dataset]),
            numerical_features=torch.stack([item[1] for item in test_dataset]),
            toured_labels=torch.stack([item[2] for item in test_dataset]),
            lead_ids=torch.stack([item[5] for item in test_dataset])
        )
        
        # Create data loaders
        train_loader = DataLoader(train_tour_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_tour_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Loaded {len(train_tour_dataset)} training samples and {len(test_tour_dataset)} testing samples")
        
        # Calculate class weights for imbalance
        toured_count = sum([item[2].item() for item in train_dataset])
        non_toured_count = len(train_dataset) - toured_count
        toured_rate = toured_count / len(train_dataset)
        
        print(f"Class distribution: {toured_count} positive ({toured_rate*100:.2f}%), {non_toured_count} negative")
        
        # Calculate class weights inversely proportional to class frequencies
        pos_weight = 1.0 / max(0.1, toured_rate)
        neg_weight = 1.0 / max(0.1, 1.0 - toured_rate)
        
        # Normalize weights to sum to 2.0
        total_weight = pos_weight + neg_weight
        pos_weight = pos_weight / total_weight * 2.0
        neg_weight = neg_weight / total_weight * 2.0
        
        print(f"Using class weights: positive={pos_weight:.4f}, negative={neg_weight:.4f}")
        
        # Calculate embedding dimensions
        embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]
        
        # Parse hidden layer dimensions
        hidden_units = [int(dim) for dim in args.hidden_layers.split(',')]
        
        # Create model
        model = TourPredictionModel(
            categorical_dims=categorical_dims,
            embedding_dims=embedding_dims,
            numerical_dim=numerical_dim,
            hidden_units=hidden_units,
            dropout=args.dropout,
            use_batch_norm=True
        ).to(device)
        
        print(f"\nModel architecture:")
        print(model)
        
        # Count parameters
        model_size = sum(p.numel() for p in model.parameters())
        print(f"Model has {model_size:,} parameters")
        
        # Create optimizer with weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        print(f"\nStarting training for {args.epochs} epochs...")
        
        # Train model
        model, history = train_tour_prediction_model(
            model=model,
            train_loader=train_loader,
            valid_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            model_save_path=os.path.join(args.output_dir, f'seed_{seed}_{args.model_name}'),
            grad_accumulation_steps=4,
            early_stopping_patience=10,
            mixed_precision=True
        )
        
        # Store results for this seed
        seed_results = {
            'seed': seed,
            'history': history,
            'model_path': os.path.join(args.output_dir, f'seed_{seed}_{args.model_name}'),
            'toured_rate': toured_rate,
            'model_size': model_size,
            'feature_names': feature_names
        }
        all_results.append(seed_results)
        
        # Save training history for this seed
        history_path = os.path.join(args.output_dir, f'seed_{seed}_training_history.joblib')
        joblib.dump(history, history_path)
        print(f"Training history saved to {history_path}")
        
        # Plot training curves for this seed
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(history['val_auc'], label='AUC')
        plt.plot(history['val_apr'], label='APR')
        plt.title('AUC & APR')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        if 'val_p_at_k' in history and 10 in history['val_p_at_k']:
            plt.plot(history['val_p_at_k'][10], label='P@10')
            plt.plot(history['val_p_at_k'][50], label='P@50')
            plt.plot(history['val_p_at_k'][100], label='P@100')
        else:
            print("Warning: precision@k metrics not found in history")
        plt.title('Precision@k')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        if 'val_p_at_k' in history:
            max_k = max(history['val_p_at_k'].keys())
            plt.plot(history['val_p_at_k'][max_k], label=f'P@{max_k}')
        plt.title('Precision@max')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'seed_{seed}_training_curves.png'))
        plt.close()
        
        print(f"\nTraining complete for seed {seed}!")
        print(f"Final model saved to {os.path.join(args.output_dir, f'seed_{seed}_{args.model_name}')}")
        
        # Feature importance using attention weights
        print("\nCalculating feature importance...")
        model.eval()
        feature_importance = torch.zeros(numerical_dim + sum(embedding_dims)).to(device)
        count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                categorical_inputs = batch[0].to(device) if batch[0].size(1) > 0 else None
                numerical_inputs = batch[1].to(device) if batch[1].size(1) > 0 else None
                
                # Skip empty batches
                if (categorical_inputs is None or categorical_inputs.size(0) == 0) and \
                   (numerical_inputs is None or numerical_inputs.size(0) == 0):
                    continue
                    
                # Get attention weights
                _, attention_weights = model(categorical_inputs, numerical_inputs)
                feature_importance += attention_weights.sum(dim=0)
                count += 1
                
                if count >= 10:  # Limit to 10 batches for speed
                    break
        
        # Normalize importance
        if count > 0:
            feature_importance = feature_importance / count
            feature_importance = feature_importance.cpu().numpy()
            
            # Plot top 20 features by importance
            plt.figure(figsize=(12, 8))
            plt.barh(range(min(20, len(feature_importance))), 
                    feature_importance[:20][::-1])
            plt.title('Feature Importance (from Attention Weights)')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'seed_{seed}_feature_importance.png'))
            plt.close()
            
            # Store feature importance in results
            seed_results['feature_importance'] = feature_importance
    
    # Save all results
    results_path = os.path.join(args.output_dir, args.output_file)
    joblib.dump(all_results, results_path)
    print(f"\nAll results saved to {results_path}")
    
    print("\nAnalysis complete for all seeds!")

if __name__ == "__main__":
    main()
