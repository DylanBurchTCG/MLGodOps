import torch
import torch.nn as nn
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
    def __init__(self, categorical_features, numerical_features, tour_labels=None, apply_labels=None, rent_labels=None,
                 lead_ids=None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.tour_labels = tour_labels
        self.apply_labels = apply_labels
        self.rent_labels = rent_labels
        self.lead_ids = lead_ids

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([]),
            self.numerical_features[idx]
        ]

        if self.tour_labels is not None:
            items.append(self.tour_labels[idx])
        if self.apply_labels is not None:
            items.append(self.apply_labels[idx])
        if self.rent_labels is not None:
            items.append(self.rent_labels[idx])
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
                tour_weight=1.0, apply_weight=1.0, rent_weight=2.0, device='cuda',
                early_stopping_patience=5, model_save_path='best_model.pt'):
    """
    Train the lead funnel model with multi-stage approach
    """
    # Loss functions - binary cross entropy for classification tasks
    tour_criterion = nn.BCELoss()
    apply_criterion = nn.BCELoss()
    rent_criterion = nn.BCELoss()

    # For ranking-specific tasks, use ranking loss
    ranking_criterion = ListMLELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'tour_auc': [], 'apply_auc': [], 'rent_auc': [],
        'tour_apr': [], 'apply_apr': [], 'rent_apr': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch in progress_bar:
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            tour_labels = batch[2].to(device)
            apply_labels = batch[3].to(device)
            rent_labels = batch[4].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            tour_pred, apply_pred, rent_pred = model(categorical_inputs, numerical_inputs)

            # Compute stage-specific losses
            tour_loss = tour_criterion(tour_pred, tour_labels)
            apply_loss = apply_criterion(apply_pred, apply_labels)
            rent_loss = rent_criterion(rent_pred, rent_labels)

            # Combined loss with weighting
            loss = tour_weight * tour_loss + apply_weight * apply_loss + rent_weight * rent_loss

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Track loss
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # Collect predictions and labels for metrics calculation
        tour_preds, tour_true = [], []
        apply_preds, apply_true = [], []
        rent_preds, rent_true = [], []

        with torch.no_grad():
            progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")
            for batch in progress_bar:
                categorical_inputs = batch[0].to(device)
                numerical_inputs = batch[1].to(device)
                tour_labels = batch[2].to(device)
                apply_labels = batch[3].to(device)
                rent_labels = batch[4].to(device)

                # Forward pass
                tour_pred, apply_pred, rent_pred = model(categorical_inputs, numerical_inputs)

                # Compute losses
                tour_loss = tour_criterion(tour_pred, tour_labels)
                apply_loss = apply_criterion(apply_pred, apply_labels)
                rent_loss = rent_criterion(rent_pred, rent_labels)

                # Combined loss
                loss = tour_weight * tour_loss + apply_weight * apply_loss + rent_weight * rent_loss
                val_loss += loss.item()

                # Collect predictions and labels
                tour_preds.append(tour_pred.cpu().numpy())
                tour_true.append(tour_labels.cpu().numpy())
                apply_preds.append(apply_pred.cpu().numpy())
                apply_true.append(apply_labels.cpu().numpy())
                rent_preds.append(rent_pred.cpu().numpy())
                rent_true.append(rent_labels.cpu().numpy())

        # Calculate average validation loss
        val_loss /= len(valid_loader)
        history['val_loss'].append(val_loss)

        # Compute metrics
        tour_preds = np.vstack(tour_preds)
        tour_true = np.vstack(tour_true)
        apply_preds = np.vstack(apply_preds)
        apply_true = np.vstack(apply_true)
        rent_preds = np.vstack(rent_preds)
        rent_true = np.vstack(rent_true)

        # AUC for each stage
        tour_auc = roc_auc_score(tour_true, tour_preds)
        apply_auc = roc_auc_score(apply_true, apply_preds)
        rent_auc = roc_auc_score(rent_true, rent_preds)

        # Average Precision for each stage
        tour_apr = average_precision_score(tour_true, tour_preds)
        apply_apr = average_precision_score(apply_true, apply_preds)
        rent_apr = average_precision_score(rent_true, rent_preds)

        # Store metrics
        history['tour_auc'].append(tour_auc)
        history['apply_auc'].append(apply_auc)
        history['rent_auc'].append(rent_auc)
        history['tour_apr'].append(tour_apr)
        history['apply_apr'].append(apply_apr)
        history['rent_apr'].append(rent_apr)

        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Tour - AUC: {tour_auc:.4f}, APR: {tour_apr:.4f}")
        print(f"Apply - AUC: {apply_auc:.4f}, APR: {apply_apr:.4f}")
        print(f"Rent - AUC: {rent_auc:.4f}, APR: {rent_apr:.4f}")

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
def rank_leads_multi_stage(model, dataloader, tour_k=500, apply_k=250, rent_k=125, device='cuda'):
    """
    Perform multi-stage ranking of leads:
    1. Rank all leads by tour probability and select top tour_k
    2. Rank those leads by apply probability and select top apply_k
    3. Rank those leads by rent probability and select top rent_k
    """
    model.eval()
    all_lead_ids = []
    all_categorical = []
    all_numerical = []
    all_tour_preds = []
    all_apply_preds = []
    all_rent_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ranking leads"):
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            lead_ids = batch[-1].cpu().numpy()

            # Get predictions for all stages
            tour_pred, apply_pred, rent_pred = model(categorical_inputs, numerical_inputs)

            # Store data
            all_lead_ids.extend(lead_ids)
            all_categorical.append(categorical_inputs.cpu())
            all_numerical.append(numerical_inputs.cpu())
            all_tour_preds.extend(tour_pred.cpu().numpy())
            all_apply_preds.extend(apply_pred.cpu().numpy())
            all_rent_preds.extend(rent_pred.cpu().numpy())

    # Convert to arrays/tensors
    all_lead_ids = np.array(all_lead_ids)
    all_tour_preds = np.array(all_tour_preds)
    all_apply_preds = np.array(all_apply_preds)
    all_rent_preds = np.array(all_rent_preds)

    # Stage 1: Rank by tour probability
    tour_indices = np.argsort(all_tour_preds.flatten())[::-1][:tour_k]
    tour_selected_ids = all_lead_ids[tour_indices]
    tour_selected_scores = all_tour_preds[tour_indices]
    tour_excluded_ids = np.setdiff1d(all_lead_ids, tour_selected_ids)

    # Stage 2: Among tour selections, rank by apply probability
    apply_indices = np.argsort(all_apply_preds[tour_indices].flatten())[::-1][:apply_k]
    apply_selected_indices = tour_indices[apply_indices]
    apply_selected_ids = all_lead_ids[apply_selected_indices]
    apply_selected_scores = all_apply_preds[tour_indices][apply_indices]
    apply_excluded_ids = np.setdiff1d(tour_selected_ids, apply_selected_ids)

    # Stage 3: Among apply selections, rank by rent probability
    rent_indices = np.argsort(all_rent_preds[apply_selected_indices].flatten())[::-1][:rent_k]
    rent_selected_indices = apply_selected_indices[rent_indices]
    rent_selected_ids = all_lead_ids[rent_selected_indices]
    rent_selected_scores = all_rent_preds[apply_selected_indices][rent_indices]
    rent_excluded_ids = np.setdiff1d(apply_selected_ids, rent_selected_ids)

    # Return selections and exclusions for each stage
    result = {
        'tour': {
            'selected': tour_selected_ids,
            'scores': tour_selected_scores,
            'excluded': tour_excluded_ids
        },
        'apply': {
            'selected': apply_selected_ids,
            'scores': apply_selected_scores,
            'excluded': apply_excluded_ids
        },
        'rent': {
            'selected': rent_selected_ids,
            'scores': rent_selected_scores,
            'excluded': rent_excluded_ids
        }
    }

    return result


# Analyze differences between selected and non-selected leads
def analyze_group_differences(dataframe, selected_ids, excluded_ids, feature_names, top_n=20):
    """
    Analyze and visualize differences between selected and non-selected leads
    """
    selected_df = dataframe[dataframe['lead_id'].isin(selected_ids)]
    excluded_df = dataframe[dataframe['lead_id'].isin(excluded_ids)]

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
                                    tour_weight=2.0, apply_weight=2.0, rent_weight=2.0,
                                    epochs=5, device='cuda'):
    """
    Fine-tune the model using a small set of external examples
    """
    # Loss functions
    tour_criterion = nn.BCELoss()
    apply_criterion = nn.BCELoss()
    rent_criterion = nn.BCELoss()

    external_loader = DataLoader(external_dataset, batch_size=len(external_dataset), shuffle=True)

    model.train()

    for epoch in range(epochs):
        for batch in external_loader:
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
            tour_labels = batch[2].to(device)
            apply_labels = batch[3].to(device)
            rent_labels = batch[4].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            tour_pred, apply_pred, rent_pred = model(categorical_inputs, numerical_inputs)

            # Compute stage-specific losses with higher weights (importance weighting)
            tour_loss = tour_weight * tour_criterion(tour_pred, tour_labels)
            apply_loss = apply_weight * apply_criterion(apply_pred, apply_labels)
            rent_loss = rent_weight * rent_criterion(rent_pred, rent_labels)

            # Combined loss with extra weighting for external examples
            loss = 3.0 * (tour_loss + apply_loss + rent_loss)  # Higher weight for external examples

            # Backpropagation
            loss.backward()

            # Optimizer step
            optimizer.step()

        print(f"Fine-tuning Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model