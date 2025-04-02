import argparse
import torch
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Import our simplified modules
# In the actual script, you would have imports like:
# from simplified_tour_model import SimplifiedTourModel
# from simplified_data_prep import prepare_tour_data_simplified
# from simplified_training import train_tour_model


def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Prospect to Tour Prediction')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the lead dataset CSV')
    parser.add_argument('--output_dir', type=str, default='./tour_model_output',
                        help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--target_toured', type=str, default='TOTAL_APPOINTMENT_COMPLETED',
                        help='Column name for toured target')
    parser.add_argument('--hidden_layers', type=str, default='256,128,64',
                        help='Comma-separated list of hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use CPU if CUDA not available
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 80)
    print("SIMPLIFIED PROSPECT TO TOUR PREDICTION MODEL")
    print("=" * 80)

    # Prepare data
    print(f"\nLoading and preprocessing data from {args.data_path}...")
    train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names = prepare_tour_data_simplified(
        data_path=args.data_path,
        target_col=args.target_toured,
        random_state=args.seed
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} testing samples")

    # Calculate class distribution
    toured_count = sum([item[2].item() for item in train_dataset])
    non_toured_count = len(train_dataset) - toured_count
    toured_rate = toured_count / len(train_dataset)

    print(f"Class distribution: {toured_count} positive ({toured_rate * 100:.2f}%), {non_toured_count} negative")

    # Calculate embedding dimensions (simple rule: min(50, (dim+1)//2))
    embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]

    # Parse hidden layer dimensions
    hidden_units = [int(dim) for dim in args.hidden_layers.split(',')]

    # Create model
    model = SimplifiedTourModel(
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

    # Train model
    model, history = train_tour_model(
        model=model,
        train_loader=train_loader,
        valid_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        focal_alpha=0.25,  # Set based on class imbalance (lower value for majority class)
        focal_gamma=2.0,  # Focus on hard examples
        model_save_path=os.path.join(args.output_dir, 'tour_prediction_model.pt'),
        early_stopping_patience=7
    )

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.joblib')
    joblib.dump(history, history_path)
    print(f"Training history saved to {history_path}")

    # Plot training curves
    plt.figure(figsize=(15, 10))

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
    plt.title('Precision@k')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))

    print("\nTraining complete!")
    print(f"Model saved to {os.path.join(args.output_dir, 'tour_prediction_model.pt')}")


if __name__ == "__main__":
    main()