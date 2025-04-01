import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score


# Custom dataset for leads
class LeadDataset(Dataset):
    def __init__(self, categorical_features, numerical_features, toured_labels=None, applied_labels=None, rented_labels=None,
                 lead_ids=None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.toured_labels = toured_labels
        self.applied_labels = applied_labels
        self.rented_labels = rented_labels
        self.lead_ids = lead_ids

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([]),
            self.numerical_features[idx]
        ]

        if self.toured_labels is not None:
            items.append(self.toured_labels[idx])
        if self.applied_labels is not None:
            items.append(self.applied_labels[idx])
        if self.rented_labels is not None:
            items.append(self.rented_labels[idx])
        if self.lead_ids is not None:
            items.append(self.lead_ids[idx])

        return tuple(items)


# Custom ranking loss function (using ListMLE approach)
class ListMLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, targets):
        # Sort targets in descending order
        sorted_targets, indices = torch.sort(targets, descending=True, dim=0)

        # Reorder scores according to target sorting
        ordered_scores = torch.gather(scores, 0, indices)

        # Calculate log softmax probabilities
        scores_softmax = F.log_softmax(ordered_scores, dim=0)

        # Compute the negative log likelihood
        loss = -torch.sum(scores_softmax)
        return loss


# Training function with multi-stage approach
def train_model(model, train_loader, valid_loader, optimizer, scheduler, num_epochs=30,
                toured_weight=1.0, applied_weight=1.0, rented_weight=2.0, device='cuda',
                early_stopping_patience=5, model_save_path='best_model.pt',
                mixed_precision=True, gradient_accumulation_steps=1):
    """
    Train the lead funnel model with multi-stage approach
    """
    # Enable mixed precision training if requested (speeds up training on newer GPUs)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None

    # Loss functions - binary cross entropy for classification tasks
    toured_criterion = nn.BCELoss()
    applied_criterion = nn.BCELoss()
    rented_criterion = nn.BCELoss()

    # For ranking-specific tasks, use ranking loss
    ranking_criterion = ListMLELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'toured_auc': [], 'applied_auc': [], 'rented_auc': [],
        'toured_apr': [], 'applied_apr': [], 'rented_apr': []
    }

    # Track GPU memory usage if using CUDA
    if device.type == 'cuda':
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero gradients once at the beginning of each epoch

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(progress_bar):
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            toured_labels = batch[2].to(device)
            applied_labels = batch[3].to(device)
            rented_labels = batch[4].to(device)

            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    toured_pred, applied_pred, rented_pred = model(categorical_inputs, numerical_inputs)

                    # Compute stage-specific losses
                    toured_loss = toured_criterion(toured_pred, toured_labels)
                    applied_loss = applied_criterion(applied_pred, applied_labels)
                    rented_loss = rented_criterion(rented_pred, rented_labels)

                    # Combined loss with weighting
                    loss = toured_weight * toured_loss + applied_weight * applied_loss + rented_weight * rented_loss
                    # Normalize loss if using gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with scaling
                scaler.scale(loss).backward()

                # Only step every gradient_accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step with scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Forward pass
                toured_pred, applied_pred, rented_pred = model(categorical_inputs, numerical_inputs)

                # Compute stage-specific losses
                toured_loss = toured_criterion(toured_pred, toured_labels)
                applied_loss = applied_criterion(applied_pred, applied_labels)
                rented_loss = rented_criterion(rented_pred, rented_labels)

                # Combined loss with weighting
                loss = toured_weight * toured_loss + applied_weight * applied_loss + rented_weight * rented_loss
                # Normalize loss if using gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backpropagation
                loss.backward()

                # Only step every gradient_accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()

            # Track loss
            train_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

            # Print memory usage every 100 batches if using CUDA
            if device.type == 'cuda' and batch_idx % 100 == 0:
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Collect predictions and labels for metrics calculation
        toured_preds, toured_true = [], []
        applied_preds, applied_true = [], []
        rented_preds, rented_true = [], []

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")
            for batch in progress_bar:
                categorical_inputs = batch[0].to(device)
                numerical_inputs = batch[1].to(device)
                toured_labels = batch[2].to(device)
                applied_labels = batch[3].to(device)
                rented_labels = batch[4].to(device)

                # Forward pass
                toured_pred, applied_pred, rented_pred = model(categorical_inputs, numerical_inputs)

                # Compute losses
                toured_loss = toured_criterion(toured_pred, toured_labels)
                applied_loss = applied_criterion(applied_pred, applied_labels)
                rented_loss = rented_criterion(rented_pred, rented_labels)

                # Combined loss
                loss = toured_weight * toured_loss + applied_weight * applied_loss + rented_weight * rented_loss
                val_loss += loss.item()

                # Collect predictions and labels
                toured_preds.append(toured_pred.cpu().numpy())
                toured_true.append(toured_labels.cpu().numpy())
                applied_preds.append(applied_pred.cpu().numpy())
                applied_true.append(applied_labels.cpu().numpy())
                rented_preds.append(rented_pred.cpu().numpy())
                rented_true.append(rented_labels.cpu().numpy())

        # Calculate average validation loss
        val_loss /= len(valid_loader)
        history['val_loss'].append(val_loss)

        # Compute metrics
        toured_preds = np.vstack(toured_preds)
        toured_true = np.vstack(toured_true)
        applied_preds = np.vstack(applied_preds)
        applied_true = np.vstack(applied_true)
        rented_preds = np.vstack(rented_preds)
        rented_true = np.vstack(rented_true)

        # AUC for each stage
        toured_auc = roc_auc_score(toured_true, toured_preds)
        applied_auc = roc_auc_score(applied_true, applied_preds)
        rented_auc = roc_auc_score(rented_true, rented_preds)

        # Average Precision for each stage
        toured_apr = average_precision_score(toured_true, toured_preds)
        applied_apr = average_precision_score(applied_true, applied_preds)
        rented_apr = average_precision_score(rented_true, rented_preds)

        # Store metrics
        history['toured_auc'].append(toured_auc)
        history['applied_auc'].append(applied_auc)
        history['rented_auc'].append(rented_auc)
        history['toured_apr'].append(toured_apr)
        history['applied_apr'].append(applied_apr)
        history['rented_apr'].append(rented_apr)

        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Toured - AUC: {toured_auc:.4f}, APR: {toured_apr:.4f}")
        print(f"Applied - AUC: {applied_auc:.4f}, APR: {applied_apr:.4f}")
        print(f"Rented - AUC: {rented_auc:.4f}, APR: {rented_apr:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Check for early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    return model, history


# Stage-wise ranking function
def rank_leads_multi_stage(model, dataloader, toured_k=500, applied_k=250, rented_k=125, device='cuda'):
    """
    Perform multi-stage ranking of leads:
    1. Rank all leads by toured probability and select top toured_k
    2. Rank those leads by applied probability and select top applied_k
    3. Rank those leads by rented probability and select top rented_k
    """
    model.eval()
    all_lead_ids = []
    all_categorical = []
    all_numerical = []
    all_toured_preds = []
    all_applied_preds = []
    all_rented_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ranking leads"):
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            lead_ids = batch[-1].cpu().numpy()

            # Get predictions for all stages
            toured_pred, applied_pred, rented_pred = model(categorical_inputs, numerical_inputs)

            # Store data
            all_lead_ids.extend(lead_ids)
            all_categorical.append(categorical_inputs.cpu())
            all_numerical.append(numerical_inputs.cpu())
            all_toured_preds.extend(toured_pred.cpu().numpy())
            all_applied_preds.extend(applied_pred.cpu().numpy())
            all_rented_preds.extend(rented_pred.cpu().numpy())

    # Convert to arrays/tensors
    all_lead_ids = np.array(all_lead_ids)
    all_toured_preds = np.array(all_toured_preds)
    all_applied_preds = np.array(all_applied_preds)
    all_rented_preds = np.array(all_rented_preds)

    # Stage 1: Rank by toured probability
    toured_indices = np.argsort(all_toured_preds.flatten())[::-1][:toured_k]
    toured_selected_ids = all_lead_ids[toured_indices]
    toured_selected_scores = all_toured_preds[toured_indices]
    toured_excluded_ids = np.setdiff1d(all_lead_ids, toured_selected_ids)

    # Stage 2: Among toured selections, rank by applied probability
    applied_indices = np.argsort(all_applied_preds[toured_indices].flatten())[::-1][:applied_k]
    applied_selected_indices = toured_indices[applied_indices]
    applied_selected_ids = all_lead_ids[applied_selected_indices]
    applied_selected_scores = all_applied_preds[toured_indices][applied_indices]
    applied_excluded_ids = np.setdiff1d(toured_selected_ids, applied_selected_ids)

    # Stage 3: Among applied selections, rank by rented probability
    rented_indices = np.argsort(all_rented_preds[applied_selected_indices].flatten())[::-1][:rented_k]
    rented_selected_indices = applied_selected_indices[rented_indices]
    rented_selected_ids = all_lead_ids[rented_selected_indices]
    rented_selected_scores = all_rented_preds[applied_selected_indices][rented_indices]
    rented_excluded_ids = np.setdiff1d(applied_selected_ids, rented_selected_ids)

    # Return selections and exclusions for each stage
    result = {
        'toured': {
            'selected': toured_selected_ids,
            'scores': toured_selected_scores,
            'excluded': toured_excluded_ids
        },
        'applied': {
            'selected': applied_selected_ids,
            'scores': applied_selected_scores,
            'excluded': applied_excluded_ids
        },
        'rented': {
            'selected': rented_selected_ids,
            'scores': rented_selected_scores,
            'excluded': rented_excluded_ids
        }
    }

    return result


# Analyze differences between selected and non-selected leads
def analyze_group_differences(dataframe, selected_ids, excluded_ids, feature_names, top_n=20):
    """
    Analyze and visualize differences between selected and non-selected leads

    Args:
        dataframe: Original DataFrame containing all leads
        selected_ids: IDs of selected leads
        excluded_ids: IDs of excluded leads
        feature_names: List of feature names to analyze
        top_n: Number of top features to return

    Returns:
        top_features: List of top N features with their differences
        differences: Dictionary of all feature differences
    """
    selected_df = dataframe[dataframe['CLIENT_PERSON_ID'].isin(selected_ids)]
    excluded_df = dataframe[dataframe['CLIENT_PERSON_ID'].isin(excluded_ids)]

    # Calculate differences for each feature
    differences = {}

    for feature in feature_names:
        if feature in dataframe.columns:
            # Calculate means for each group
            if dataframe[feature].dtype in [np.float64, np.int64]:
                selected_mean = selected_df[feature].mean()
                excluded_mean = excluded_df[feature].mean()

                # Calculate differences
                abs_diff = selected_mean - excluded_mean
                if excluded_mean != 0:
                    pct_diff = (abs_diff / excluded_mean) * 100
                else:
                    pct_diff = np.nan

                # Store results
                differences[feature] = {
                    'selected_mean': selected_mean,
                    'excluded_mean': excluded_mean,
                    'abs_diff': abs_diff,
                    'pct_diff': pct_diff
                }

    # Sort by absolute difference
    sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]['abs_diff']), reverse=True)

    # Create visualization for top N features
    top_features = sorted_diffs[:top_n]

    # Return for further analysis
    return top_features, differences


# Incorporate external examples for fine-tuning
def finetune_with_external_examples(model, external_dataset, optimizer,
                                    toured_weight=2.0, applied_weight=2.0, rented_weight=2.0,
                                    epochs=5, device='cuda'):
    """
    Fine-tune the model using a small set of external examples

    Args:
        model: The trained model to fine-tune
        external_dataset: Dataset containing external examples
        optimizer: Optimizer to use for fine-tuning
        toured_weight: Weight for toured stage loss
        applied_weight: Weight for applied stage loss
        rented_weight: Weight for rented stage loss
        epochs: Number of fine-tuning epochs
        device: Device to use (cuda or cpu)

    Returns:
        model: Fine-tuned model
    """
    # Loss functions
    toured_criterion = nn.BCELoss()
    applied_criterion = nn.BCELoss()
    rented_criterion = nn.BCELoss()

    external_loader = DataLoader(external_dataset, batch_size=len(external_dataset), shuffle=True)

    model.train()

    for epoch in range(epochs):
        for batch in external_loader:
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            toured_labels = batch[2].to(device)
            applied_labels = batch[3].to(device)
            rented_labels = batch[4].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            toured_pred, applied_pred, rented_pred = model(categorical_inputs, numerical_inputs)

            # Compute stage-specific losses with higher weights (importance weighting)
            toured_loss = toured_weight * toured_criterion(toured_pred, toured_labels)
            applied_loss = applied_weight * applied_criterion(applied_pred, applied_labels)
            rented_loss = rented_weight * rented_criterion(rented_pred, rented_labels)

            # Combined loss with extra weighting for external examples
            loss = 3.0 * (toured_loss + applied_loss + rented_loss)  # Higher weight for external examples

            # Backpropagation
            loss.backward()

            # Optimizer step
            optimizer.step()

        print(f"Fine-tuning Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model