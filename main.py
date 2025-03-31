import torch
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Import our modules
from model_architecture import LeadFunnelModel
from training_pipeline import train_model, rank_leads_multi_stage, analyze_group_differences, \
    finetune_with_external_examples
from data_preparation import preprocess_data, prepare_external_examples, visualize_group_differences


def parse_args():
    parser = argparse.ArgumentParser(description='Lead Funnel Prediction Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the lead dataset CSV')
    parser.add_argument('--external_data_path', type=str, help='Path to external examples CSV for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--tour_k', type=int, default=500, help='Number of leads to select at tour stage')
    parser.add_argument('--apply_k', type=int, default=250, help='Number of leads to select at apply stage')
    parser.add_argument('--rent_k', type=int, default=125, help='Number of leads to select at rent stage')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Choose device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Preprocess the data
    train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names = preprocess_data(
        args.data_path,
        save_preprocessors=True,
        preprocessors_path=os.path.join(args.output_dir, 'preprocessors')
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate embedding dimensions based on cardinality
    embedding_dims = [min(50, (dim + 1) // 2) for dim in categorical_dims]

    # Initialize model
    model = LeadFunnelModel(
        categorical_dims=categorical_dims,
        embedding_dims=embedding_dims,
        numerical_dim=numerical_dim,
        transformer_dim=256,  # Dimensionality of transformer
        num_heads=4,  # Number of attention heads
        num_transformer_blocks=3,  # Number of transformer layers
        ff_dim=512,  # Feed-forward dimension in transformer
        head_hidden_dims=[128, 64],  # Hidden dimensions in prediction heads
        dropout=0.2
    ).to(device)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Train the model
    model, history = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        tour_weight=1.0,
        apply_weight=1.0,
        rent_weight=2.0,  # Higher weight for rent prediction
        device=device,
        model_save_path=os.path.join(args.output_dir, 'best_model.pt')
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
            tour_weight=2.0,
            apply_weight=2.0,
            rent_weight=2.0,
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
        tour_k=args.tour_k,
        apply_k=args.apply_k,
        rent_k=args.rent_k,
        device=device
    )

    # Save rankings
    for stage in ['tour', 'apply', 'rent']:
        pd.DataFrame({
            'lead_id': rankings[stage]['selected'],
            'score': rankings[stage]['scores'].flatten()
        }).to_csv(os.path.join(args.output_dir, f'{stage}_selected_leads.csv'), index=False)

    # Load the original data for analysis
    full_data = pd.read_csv(args.data_path)

    # Analyze and visualize differences between selected and excluded leads at each stage
    for stage in ['tour', 'apply', 'rent']:
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
    plt.plot(history['tour_auc'], label='Tour')
    plt.plot(history['apply_auc'], label='Apply')
    plt.plot(history['rent_auc'], label='Rent')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['tour_apr'], label='Tour')
    plt.plot(history['apply_apr'], label='Apply')
    plt.plot(history['rent_apr'], label='Rent')
    plt.title('Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'learning_curves.png'))

    # Print final performance summary
    print("\nFinal Performance Summary:")
    print(f"Tour - AUC: {history['tour_auc'][-1]:.4f}, APR: {history['tour_apr'][-1]:.4f}")
    print(f"Apply - AUC: {history['apply_auc'][-1]:.4f}, APR: {history['apply_apr'][-1]:.4f}")
    print(f"Rent - AUC: {history['rent_auc'][-1]:.4f}, APR: {history['rent_apr'][-1]:.4f}")

    print("\nModel training and evaluation complete!")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()