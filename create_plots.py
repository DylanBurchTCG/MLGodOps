import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Create plots from model results')
    parser.add_argument('--result_file', type=str, required=True, help='Path to results joblib file')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--subset', type=str, default=None, help='Specific subset to plot (e.g., "subset1")')
    parser.add_argument('--seed', type=str, default=None, help='Specific seed to plot (e.g., "seed_42")')
    return parser.parse_args()


def plot_metrics(results, output_dir, subset_filter=None, seed_filter=None):
    """Create ROC and PR curve plots from results"""
    os.makedirs(output_dir, exist_ok=True)

    # Handle consolidated multiple seeds
    if any(key.startswith('seed_') for key in results.keys()):
        if seed_filter:
            # Plot only the specific seed
            if seed_filter in results:
                print(f"Creating plots for seed: {seed_filter}")
                plot_single_result(results[seed_filter], output_dir, subset_filter, seed_name=seed_filter)
            else:
                print(f"Seed {seed_filter} not found in results")
        else:
            # Plot all seeds
            for seed_name, seed_results in results.items():
                print(f"Creating plots for {seed_name}")
                plot_single_result(seed_results, output_dir, subset_filter, seed_name=seed_name)
    else:
        # Single seed results
        plot_single_result(results, output_dir, subset_filter)


def plot_single_result(results, output_dir, subset_filter=None, seed_name=None):
    """Plot metrics for a single result set (one seed)"""
    prefix = f"{seed_name}_" if seed_name else ""

    # Plot training history
    plot_training_history(results, output_dir, subset_filter, prefix)

    # Plot ROC and PR curves
    for subset_name, subset_data in results.items():
        if subset_name == 'metadata' or subset_name == 'preprocessing':
            continue

        if subset_filter and subset_name != subset_filter:
            continue

        print(f"  Processing {subset_name}")
        if 'metrics' in subset_data:
            metrics = subset_data['metrics']

            # Create ROC curve plot
            plt.figure(figsize=(10, 8))
            for stage in ['toured', 'applied', 'rented']:
                roc_auc = metrics[stage]['roc_auc']
                label = f"{stage.capitalize()} (AUC = {roc_auc:.4f})"
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'ROC Curve - {subset_name}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.grid(True, alpha=0.3)
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}{subset_name}_roc_curve.png"))
            plt.close()

            # Create PR curve plot
            plt.figure(figsize=(10, 8))
            for stage in ['toured', 'applied', 'rented']:
                apr = metrics[stage]['apr']
                label = f"{stage.capitalize()} (APR = {apr:.4f})"
                plt.title(f'Precision-Recall Curve - {subset_name}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.grid(True, alpha=0.3)
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}{subset_name}_pr_curve.png"))
            plt.close()
        else:
            print(f"  No metrics found for {subset_name}")


def plot_training_history(results, output_dir, subset_filter=None, prefix=""):
    """Plot training history for all subsets"""
    # Collect loss history from all subsets
    subsets_to_plot = {}
    for subset_name, subset_data in results.items():
        if subset_name == 'metadata' or subset_name == 'preprocessing':
            continue

        if subset_filter and subset_name != subset_filter:
            continue

        if 'history' in subset_data:
            subsets_to_plot[subset_name] = subset_data['history']

    if not subsets_to_plot:
        print("No training history found to plot")
        return

    # Plot training/validation loss
    plt.figure(figsize=(12, 8))
    for subset_name, history in subsets_to_plot.items():
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            plt.plot(epochs, history['train_loss'], '-', label=f'{subset_name} - Train')
            plt.plot(epochs, history['val_loss'], '--', label=f'{subset_name} - Val')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}training_loss.png"))
    plt.close()

    # Plot metrics if available
    for metric_name in ['toured_auc', 'applied_auc', 'rented_auc', 'toured_apr', 'applied_apr', 'rented_apr']:
        plt.figure(figsize=(12, 8))
        has_data = False

        for subset_name, history in subsets_to_plot.items():
            if metric_name in history and len(history[metric_name]) > 0:
                has_data = True
                epochs = range(1, len(history[metric_name]) + 1)
                plt.plot(epochs, history[metric_name], '-o', label=f'{subset_name}')

        if has_data:
            plt.title(f'{metric_name.replace("_", " ").title()}')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}{metric_name}.png"))
        plt.close()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading results from {args.result_file}")
    results = joblib.load(args.result_file)

    plot_metrics(results, args.output_dir, args.subset, args.seed)
    print(f"Plots saved to {args.output_dir}")