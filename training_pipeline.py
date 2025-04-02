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
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
import csv

# Import FocalLoss from model_architecture
from model_architecture import FocalLoss


# ----------------
# Dataset Class
# ----------------
class LeadDataset(Dataset):
    def __init__(self, categorical_features, numerical_features,
                 toured_labels=None, applied_labels=None, rented_labels=None,
                 lead_ids=None):
        # Ensure categorical features are long type for embedding lookup
        if categorical_features is not None and categorical_features.dtype != torch.long:
            self.categorical_features = categorical_features.long()
        else:
            self.categorical_features = categorical_features
            
        # Ensure numerical features are float type
        if numerical_features is not None and numerical_features.dtype != torch.float32:
            self.numerical_features = numerical_features.float()
        else:
            self.numerical_features = numerical_features
            
        self.toured_labels = toured_labels
        self.applied_labels = applied_labels
        self.rented_labels = rented_labels
        self.lead_ids = lead_ids

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([], dtype=torch.long),
            self.numerical_features[idx]
        ]

        # Add labels if present
        if self.toured_labels is not None:
            items.append(self.toured_labels[idx])
        if self.applied_labels is not None:
            items.append(self.applied_labels[idx])
        if self.rented_labels is not None:
            items.append(self.rented_labels[idx])

        # Add lead_id if present
        if self.lead_ids is not None:
            items.append(self.lead_ids[idx])

        return tuple(items)


# ----------------
# Multi-Stage Ranking
# ----------------
def rank_leads_multi_stage(model, dataloader,
                           toured_k=2000, applied_k=1000, rented_k=250,
                           use_percentages=False, toured_pct=0.5, applied_pct=0.5, rented_pct=0.5,
                           device='cuda'):
    """
    Ranks leads in a funnel:
     1) Sort by Toured prob => top K or top percentage
     2) Among them, sort by Applied prob => top K or top percentage
     3) Among them, sort by Rented prob => top K or top percentage
    Returns a dict with selected IDs and excluded IDs for each stage.
    
    Args:
        model: The trained model
        dataloader: DataLoader containing the leads
        toured_k, applied_k, rented_k: Fixed counts for selection at each stage
        use_percentages: Whether to use percentage-based selection instead of fixed counts
        toured_pct, applied_pct, rented_pct: Percentages to use for selection at each stage
        device: Device to use for prediction
    """
    model.eval()

    all_lead_ids = []
    all_toured_preds = []
    all_applied_preds = []
    all_rented_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ranking leads"):
            cat_in = batch[0].to(device).long()  # Ensure long type for embedding
            num_in = batch[1].to(device).float()  # Ensure float type for numerical
            lead_ids = batch[-1].cpu().numpy()

            # Skip empty batches
            if cat_in.size(0) == 0:
                continue

            try:
                # Predict - support both standard and cascaded model interfaces
                outputs = model(cat_in, num_in)
                if isinstance(outputs, tuple) and len(outputs) >= 3:
                    # Extract only predictions (first 3 outputs)
                    toured_pred, applied_pred, rented_pred = outputs[:3]
                else:
                    toured_pred, applied_pred, rented_pred = outputs

                all_lead_ids.extend(lead_ids)
                all_toured_preds.extend(toured_pred.cpu().numpy())
                all_applied_preds.extend(applied_pred.cpu().numpy())
                all_rented_preds.extend(rented_pred.cpu().numpy())
            except Exception as e:
                print(f"Error in ranking batch: {str(e)}")
                continue

    # Convert to arrays
    all_lead_ids = np.array(all_lead_ids)
    all_toured_preds = np.array(all_toured_preds).flatten()
    all_applied_preds = np.array(all_applied_preds).flatten()
    all_rented_preds = np.array(all_rented_preds).flatten()

    # Handle empty results case
    if len(all_lead_ids) == 0:
        print("Warning: No leads to rank")
        return {
            'toured': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])},
            'applied': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])},
            'rented': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])}
        }

    # Stage 1: Toured - determine selection count
    if use_percentages:
        # Use percentage-based selection
        toured_k = max(1, int(len(all_lead_ids) * toured_pct))
        print(f"Using {toured_pct*100:.1f}% for toured stage: {toured_k} leads")
    else:
        # Use fixed count selection (with safety check)
        toured_k = min(toured_k, len(all_lead_ids))
        print(f"Using fixed count for toured stage: {toured_k} leads")

    # Stage 1: Toured
    toured_indices = np.argsort(all_toured_preds)[::-1][:toured_k]
    toured_selected_ids = all_lead_ids[toured_indices]
    toured_scores = all_toured_preds[toured_indices]
    toured_excluded_ids = np.setdiff1d(all_lead_ids, toured_selected_ids)

    print(f"Selected {len(toured_selected_ids)} out of {len(all_lead_ids)} leads for touring")

    # Stage 2: Applied - determine selection count
    if use_percentages:
        # Use percentage-based selection
        applied_k = max(1, int(len(toured_selected_ids) * applied_pct))
        print(f"Using {applied_pct*100:.1f}% for applied stage: {applied_k} leads")
    else:
        # Use fixed count selection (with safety check)
        applied_k = min(applied_k, len(toured_selected_ids))
        print(f"Using fixed count for applied stage: {applied_k} leads")

    # Stage 2: Among the top Toured, rank by Applied
    applied_subset = all_applied_preds[toured_indices]
    applied_indices = np.argsort(applied_subset)[::-1][:applied_k]
    applied_selected_ids = toured_selected_ids[applied_indices]
    applied_scores = applied_subset[applied_indices]
    applied_excluded_ids = np.setdiff1d(toured_selected_ids, applied_selected_ids)

    print(f"Selected {len(applied_selected_ids)} out of {len(toured_selected_ids)} leads for applications")

    # Stage 3: Rented - determine selection count
    if use_percentages:
        # Use percentage-based selection
        rented_k = max(1, int(len(applied_selected_ids) * rented_pct))
        print(f"Using {rented_pct*100:.1f}% for rented stage: {rented_k} leads")
    else:
        # Use fixed count selection (with safety check)
        rented_k = min(rented_k, len(applied_selected_ids))
        print(f"Using fixed count for rented stage: {rented_k} leads")

    # Stage 3: Among top Applied, rank by Rented
    rented_subset = all_rented_preds[toured_indices][applied_indices]
    rented_indices = np.argsort(rented_subset)[::-1][:rented_k]
    rented_selected_ids = applied_selected_ids[rented_indices]
    rented_scores = rented_subset[rented_indices]
    rented_excluded_ids = np.setdiff1d(applied_selected_ids, rented_selected_ids)

    print(f"Selected {len(rented_selected_ids)} out of {len(applied_selected_ids)} leads for rental")

    result = {
        'toured': {
            'selected': toured_selected_ids,
            'scores': toured_scores,
            'excluded': toured_excluded_ids
        },
        'applied': {
            'selected': applied_selected_ids,
            'scores': applied_scores,
            'excluded': applied_excluded_ids
        },
        'rented': {
            'selected': rented_selected_ids,
            'scores': rented_scores,
            'excluded': rented_excluded_ids
        }
    }
    return result


# ----------------
# Group Differences
# ----------------
def analyze_group_differences(dataframe, selected_ids, excluded_ids, feature_names, top_n=20):
    """
    For interpretability: compare average numeric feature values
    between selected vs. excluded groups, returning the top absolute differences.
    """
    selected_df = dataframe[dataframe['CLIENT_PERSON_ID'].isin(selected_ids)]
    excluded_df = dataframe[dataframe['CLIENT_PERSON_ID'].isin(excluded_ids)]

    differences = {}
    for feat in feature_names:
        if feat in dataframe.columns:
            # Only do numeric columns
            if pd.api.types.is_numeric_dtype(dataframe[feat]):
                sel_mean = selected_df[feat].mean()
                exc_mean = excluded_df[feat].mean()
                abs_diff = sel_mean - exc_mean
                pct_diff = np.nan
                if exc_mean != 0:
                    pct_diff = abs_diff / exc_mean * 100.0

                differences[feat] = {
                    'selected_mean': sel_mean,
                    'excluded_mean': exc_mean,
                    'abs_diff': abs_diff,
                    'pct_diff': pct_diff
                }

    # Sort by absolute difference
    sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]['abs_diff']), reverse=True)
    top_features = sorted_diffs[:top_n]
    return top_features, differences


# ----------------
# Fine-tune with Extra Data
# ----------------
def finetune_with_external_examples(model,
                                    external_dataset,
                                    optimizer,
                                    toured_weight=2.0,
                                    applied_weight=2.0,
                                    rented_weight=2.0,
                                    ranking_weight=0.3,
                                    epochs=5,
                                    device='cuda'):
    """
    Optionally incorporate new examples or corrected data
    by running a short fine-tuning loop.
    """
    # Use FocalLoss instead of BCELoss for consistency
    # Set alpha appropriately (assuming external data might also be imbalanced)
    toured_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    applied_criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')
    rented_criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')

    # Add ranking loss
    ranking_criterion = nn.BCELoss()  # placeholder, will use model's ranking loss if available

    # Use a more practical batch size for fine-tuning instead of loading all at once
    batch_size = min(256, len(external_dataset))
    ext_loader = DataLoader(external_dataset, batch_size=batch_size, shuffle=True)

    print(f"Fine-tuning with {len(external_dataset)} examples using batch size {batch_size}")

    model.train()

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
            outputs = model(cat_in, num_in, lead_ids)

            # Support both standard and cascaded model interfaces
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                # Extract only predictions (first 3 outputs)
                toured_pred, applied_pred, rented_pred = outputs[:3]
            else:
                toured_pred, applied_pred, rented_pred = outputs

            # Standard BCE loss with gradient clipping
            toured_loss = toured_weight * toured_criterion(toured_pred, toured_labels)
            applied_loss = applied_weight * applied_criterion(applied_pred, applied_labels)
            rented_loss = rented_weight * rented_criterion(rented_pred, rented_labels)

            # Scale for fine-tuning
            loss = 3.0 * (toured_loss + applied_loss + rented_loss)

            # Check for invalid loss values
            if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
                print(f"Warning: Invalid loss value detected: {loss.item()}. Skipping batch.")
                continue

            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        print(f"Fine-tuning Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    return model


# ----------------
# Train the Model with Precision@k tracking
# ----------------
def train_model(model,
                train_loader,
                valid_loader,
                optimizer,
                scheduler,
                num_epochs=30,
                toured_weight=1.0,
                applied_weight=1.0,
                rented_weight=2.0,
                ranking_weight=0.3,  # Added ranking loss weight
                device='cuda',
                early_stopping_patience=5,
                model_save_path='best_model.pt',
                mixed_precision=True,
                gradient_accumulation_steps=1):
    """
    Standard training loop for multi-stage model (toured, applied, rented).
    Uses BCE loss for each stage, plus weighting and optionally ranking loss.
    """
    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device == 'cuda') else None

    # Stage-specific BCE loss
    toured_criterion = nn.BCELoss(reduction='mean')
    applied_criterion = nn.BCELoss(reduction='mean')
    rented_criterion = nn.BCELoss(reduction='mean')

    # Add ranking loss if the model supports it
    ranking_criterion = None
    if hasattr(model, 'ranking_loss'):
        ranking_criterion = model.ranking_loss

    best_val_loss = float('inf')
    early_stopping_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'toured_auc': [], 'applied_auc': [], 'rented_auc': [],
        'toured_apr': [], 'applied_apr': [], 'rented_apr': [],
        'toured_p10': [], 'applied_p10': [], 'rented_p10': [],  # Added Precision@10
        'toured_p50': [], 'applied_p50': [], 'rented_p50': [],  # Added Precision@50
        'toured_pmax': [], 'applied_pmax': [], 'rented_pmax': []  # Added Precision@max
    }

    print(f"\nStarting training for {num_epochs} epochs with device={device}")
    print(f"Mixed precision: {mixed_precision}, Grad accumulation: {gradient_accumulation_steps}")
    print(f"Loss weights: toured={toured_weight}, applied={applied_weight}, rented={rented_weight}")
    
    for epoch in range(num_epochs):
        # -------- TRAINING --------
        model.train()
        train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_iter):
            categorical_inputs = batch[0].to(device).long()  # Ensure long type for embedding
            numerical_inputs = batch[1].to(device).float()  # Ensure float type for numerical
            toured_labels = batch[2].to(device)
            applied_labels = batch[3].to(device)
            rented_labels = batch[4].to(device)
            lead_ids = batch[5].to(device) if len(batch) > 5 else None

            # Skip empty batches
            if categorical_inputs.size(0) == 0:
                continue

            try:
                if scaler is not None:
                    # Mixed precision
                    with torch.cuda.amp.autocast():
                        # Support both standard and cascaded model interfaces
                        outputs = model(categorical_inputs, numerical_inputs, lead_ids)
                        if isinstance(outputs, tuple) and len(outputs) >= 3:
                            # Extract only predictions (first 3 outputs)
                            toured_pred, applied_pred, rented_pred = outputs[:3]
                        else:
                            toured_pred, applied_pred, rented_pred = outputs

                        # Add epsilon to prevent exact 0s and 1s which can cause numerical instability
                        epsilon = 1e-7
                        toured_pred = torch.clamp(toured_pred, epsilon, 1 - epsilon)
                        applied_pred = torch.clamp(applied_pred, epsilon, 1 - epsilon)
                        rented_pred = torch.clamp(rented_pred, epsilon, 1 - epsilon)

                        # Standard BCE loss - clamped to prevent extreme values
                        toured_loss = toured_criterion(toured_pred, toured_labels)
                        applied_loss = applied_criterion(applied_pred, applied_labels)
                        rented_loss = rented_criterion(rented_pred, rented_labels)

                        # Check for NaN/Inf with strict replacement
                        if torch.isnan(toured_loss) or torch.isinf(toured_loss) or toured_loss > 10:
                            toured_loss = torch.tensor(0.5, device=device, requires_grad=True)
                            print(f"Warning: Replaced extreme toured loss")

                        if torch.isnan(applied_loss) or torch.isinf(applied_loss) or applied_loss > 10:
                            applied_loss = torch.tensor(0.5, device=device, requires_grad=True)
                            print(f"Warning: Replaced extreme applied loss")

                        if torch.isnan(rented_loss) or torch.isinf(rented_loss) or rented_loss > 10:
                            rented_loss = torch.tensor(0.5, device=device, requires_grad=True)
                            print(f"Warning: Replaced extreme rented loss")

                        # Combine loss with weighting
                        bce_loss = toured_weight * toured_loss + \
                                   applied_weight * applied_loss + \
                                   rented_weight * rented_loss

                        # Add ranking loss if model supports it and if batch size is sufficient
                        rank_loss = 0
                        if ranking_criterion and categorical_inputs.size(0) >= 10:
                            # Try to use ranking loss if batch size is large enough
                            try:
                                # Create a ranking loss for each stage
                                from model_architecture import ListMLELoss
                                ranking_module = ListMLELoss()

                                toured_ranking_loss = ranking_module(toured_pred.squeeze(), toured_labels.squeeze())
                                applied_ranking_loss = ranking_module(applied_pred.squeeze(), applied_labels.squeeze())
                                rented_ranking_loss = ranking_module(rented_pred.squeeze(), rented_labels.squeeze())

                                # Clamp to prevent extreme values
                                toured_ranking_loss = torch.clamp(toured_ranking_loss, -5, 5)
                                applied_ranking_loss = torch.clamp(applied_ranking_loss, -5, 5)
                                rented_ranking_loss = torch.clamp(rented_ranking_loss, -5, 5)

                                rank_loss = toured_weight * toured_ranking_loss + \
                                            applied_weight * applied_ranking_loss + \
                                            rented_weight * rented_ranking_loss

                                # Combine BCE and ranking loss
                                loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * rank_loss
                            except Exception as e:
                                # If ranking loss fails, just use BCE
                                print(f"Ranking loss failed: {str(e)}, using BCE only")
                                loss = bce_loss
                        else:
                            loss = bce_loss

                loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stricter gradient clipping
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                print(f"Batch shapes: cat={categorical_inputs.shape}, num={numerical_inputs.shape}")
                # Skip this batch and continue
                continue

            train_loss += loss.item() * gradient_accumulation_steps
            train_iter.set_postfix({'loss': float(loss.item() * gradient_accumulation_steps)})
            num_batches += 1

        train_loss /= max(1, num_batches)
        history['train_loss'].append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        toured_preds, toured_true = [], []
        applied_preds, applied_true = [], []
        rented_preds, rented_true = [], []

        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")
        for batch in valid_iter:
            categorical_inputs = batch[0].to(device).long()  # Ensure long type for embedding
            numerical_inputs = batch[1].to(device).float()  # Ensure float type for numerical
            toured_labels = batch[2].to(device)
            applied_labels = batch[3].to(device)
            rented_labels = batch[4].to(device)
            lead_ids = batch[5].to(device) if len(batch) > 5 else None

            # Skip empty batches
            if categorical_inputs.size(0) == 0:
                continue

            try:
                with torch.no_grad():
                    outputs = model(categorical_inputs, numerical_inputs, lead_ids)
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        # Extract only predictions (first 3 outputs)
                        toured_pred, applied_pred, rented_pred = outputs[:3]
                    else:
                        toured_pred, applied_pred, rented_pred = outputs

                    toured_loss = toured_criterion(toured_pred, toured_labels)
                    applied_loss = applied_criterion(applied_pred, applied_labels)
                    rented_loss = rented_criterion(rented_pred, rented_labels)

                    # Handle extreme values - stricter clipping to prevent huge loss values
                    if not torch.isfinite(toured_loss) or toured_loss > 50:
                        toured_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    if not torch.isfinite(applied_loss) or applied_loss > 50:
                        applied_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    if not torch.isfinite(rented_loss) or rented_loss > 50:
                        rented_loss = torch.tensor(1.0, device=device, requires_grad=True)

                    batch_loss = (toured_weight * toured_loss +
                                  applied_weight * applied_loss +
                                  rented_weight * rented_loss)

                    # Stricter clipping for batch loss
                    if not torch.isfinite(batch_loss) or batch_loss > 50:
                        batch_loss = torch.tensor(3.0, device=device, requires_grad=True)

                    val_loss += batch_loss.item()

                    toured_preds.append(toured_pred.cpu().numpy())
                    toured_true.append(toured_labels.cpu().numpy())

                    applied_preds.append(applied_pred.cpu().numpy())
                    applied_true.append(applied_labels.cpu().numpy())

                    rented_preds.append(rented_pred.cpu().numpy())
                    rented_true.append(rented_labels.cpu().numpy())
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                # Skip this batch and continue
                continue

        val_loss /= max(1, len(valid_loader))
        history['val_loss'].append(val_loss)

        # Flatten predictions
        toured_preds = np.vstack(toured_preds)
        toured_true = np.vstack(toured_true)
        applied_preds = np.vstack(applied_preds)
        applied_true = np.vstack(applied_true)
        rented_preds = np.vstack(rented_preds)
        rented_true = np.vstack(rented_true)

        # AUC + APR
        toured_auc = roc_auc_score(toured_true, toured_preds)
        applied_auc = roc_auc_score(applied_true, applied_preds)
        rented_auc = roc_auc_score(rented_true, rented_preds)

        toured_apr = average_precision_score(toured_true, toured_preds)
        applied_apr = average_precision_score(applied_true, applied_preds)
        rented_apr = average_precision_score(rented_true, rented_preds)

        # Added Precision@k calculation
        from model_architecture import precision_at_k

        # Precision@10
        k10 = min(10, len(toured_true))
        toured_p10 = precision_at_k(toured_true.flatten(), toured_preds.flatten(), k10)
        applied_p10 = precision_at_k(applied_true.flatten(), applied_preds.flatten(), k10)
        rented_p10 = precision_at_k(rented_true.flatten(), rented_preds.flatten(), k10)

        # Precision@50
        k50 = min(50, len(toured_true))
        toured_p50 = precision_at_k(toured_true.flatten(), toured_preds.flatten(), k50)
        applied_p50 = precision_at_k(applied_true.flatten(), applied_preds.flatten(), k50)
        rented_p50 = precision_at_k(rented_true.flatten(), rented_preds.flatten(), k50)

        # Precision@max (using all examples)
        k_max = len(toured_true)
        toured_pmax = precision_at_k(toured_true.flatten(), toured_preds.flatten(), k_max)
        applied_pmax = precision_at_k(applied_true.flatten(), applied_preds.flatten(), k_max)
        rented_pmax = precision_at_k(rented_true.flatten(), rented_preds.flatten(), k_max)

        # Store all metrics
        history['toured_auc'].append(toured_auc)
        history['applied_auc'].append(applied_auc)
        history['rented_auc'].append(rented_auc)
        history['toured_apr'].append(toured_apr)
        history['applied_apr'].append(applied_apr)
        history['rented_apr'].append(rented_apr)
        history['toured_p10'].append(toured_p10)
        history['applied_p10'].append(applied_p10)
        history['rented_p10'].append(rented_p10)
        history['toured_p50'].append(toured_p50)
        history['applied_p50'].append(applied_p50)
        history['rented_p50'].append(rented_p50)
        history['toured_pmax'].append(toured_pmax)
        history['applied_pmax'].append(applied_pmax)
        history['rented_pmax'].append(rented_pmax)

        # Print metrics with clear formatting and line breaks
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("\nMetrics:")
        print(f"{'Stage':<10} {'AUC':<10} {'APR':<10} {'P@10':<10} {'P@max':<10}")
        print("-" * 80)
        print(f"{'Toured':<10} {toured_auc:.4f}     {toured_apr:.4f}     {toured_p10:.4f}     {toured_pmax:.4f}")
        print(f"{'Applied':<10} {applied_auc:.4f}     {applied_apr:.4f}     {applied_p10:.4f}     {applied_pmax:.4f}")
        print(f"{'Rented':<10} {rented_auc:.4f}     {rented_apr:.4f}     {rented_p10:.4f}     {rented_pmax:.4f}")
        print(f"Max predictions: {k_max}")
        print("-" * 80 + "\n")

        # Step LR scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best state
        try:
            model.load_state_dict(torch.load(model_save_path))
        except Exception as e:
            print(f"Error loading best model state: {str(e)}")
            print("Continuing with current model state...")

    return model, history


# ----------------
# Stage-Specific Clustering
# ----------------
def perform_stage_specific_clustering(dataloader, model, device, 
                                     n_clusters=10, 
                                     stage='toured',
                                     feature_extraction_layer=None,
                                     use_umap=True,
                                     random_state=42):
    """
    Perform clustering specifically on leads that have reached a certain stage.
    This provides better insights into lead characteristics at each funnel stage.
    
    Args:
        dataloader: DataLoader containing the leads
        model: Trained model
        device: Device to use for prediction
        n_clusters: Number of clusters to create
        stage: Which stage to analyze ('toured', 'applied', 'rented')
        feature_extraction_layer: Optional hook to extract intermediate features
        use_umap: Whether to use UMAP dimensionality reduction before clustering
        
    Returns:
        Dictionary with cluster assignments, centroids, and visualizations
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    
    # First, use the model to get predictions and select leads for the stage
    model.eval()
    all_lead_ids = []
    all_features = []
    all_labels = {'toured': [], 'applied': [], 'rented': []}
    all_preds = {'toured': [], 'applied': [], 'rented': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting data for {stage} clustering"):
            cat_in = batch[0].to(device).long()  # Ensure long type for embedding
            num_in = batch[1].to(device).float()  # Ensure float type for numerical
            toured_labels = batch[2].cpu().numpy()
            applied_labels = batch[3].cpu().numpy()
            rented_labels = batch[4].cpu().numpy()
            lead_ids = batch[5].cpu().numpy() if len(batch) > 5 else np.arange(len(cat_in))
            
            # Forward pass through model 
            outputs = model(cat_in, num_in)
            
            # Get stage predictions
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                toured_pred, applied_pred, rented_pred = outputs[:3]
            else:
                toured_pred, applied_pred, rented_pred = outputs
                
            # Extract embeddings if a feature extraction layer is provided
            if feature_extraction_layer is not None:
                # This would need a hook mechanism to extract features from the model
                features = feature_extraction_layer.cpu().numpy()
            else:
                # Use concatenated inputs as features if no specific layer provided
                features = torch.cat([cat_in, num_in], dim=1).cpu().numpy()
            
            # Store everything
            all_lead_ids.extend(lead_ids)
            all_features.extend(features)
            all_labels['toured'].extend(toured_labels)
            all_labels['applied'].extend(applied_labels)
            all_labels['rented'].extend(rented_labels)
            all_preds['toured'].extend(toured_pred.cpu().numpy())
            all_preds['applied'].extend(applied_pred.cpu().numpy())
            all_preds['rented'].extend(rented_pred.cpu().numpy())
    
    # Convert to arrays
    all_lead_ids = np.array(all_lead_ids)
    all_features = np.array(all_features)
    for s in all_labels:
        all_labels[s] = np.array(all_labels[s])
        all_preds[s] = np.array(all_preds[s])
    
    # Now apply stage-specific filtering
    if stage == 'toured':
        # Use actual toured leads or predicted ones
        lead_mask = all_labels['toured'].flatten() > 0
        stage_leads = all_lead_ids[lead_mask]
        stage_features = all_features[lead_mask]
        stage_labels = all_labels['toured'][lead_mask]
        stage_preds = all_preds['toured'][lead_mask]
        print(f"Selected {len(stage_leads)} leads with actual toured=1 for clustering")
    elif stage == 'applied':
        # Use actual applied leads or predicted ones
        lead_mask = all_labels['applied'].flatten() > 0
        stage_leads = all_lead_ids[lead_mask]
        stage_features = all_features[lead_mask]
        stage_labels = all_labels['applied'][lead_mask]
        stage_preds = all_preds['applied'][lead_mask]
        print(f"Selected {len(stage_leads)} leads with actual applied=1 for clustering")
    elif stage == 'rented':
        # Use actual rented leads or predicted ones
        lead_mask = all_labels['rented'].flatten() > 0
        stage_leads = all_lead_ids[lead_mask]
        stage_features = all_features[lead_mask]
        stage_labels = all_labels['rented'][lead_mask]
        stage_preds = all_preds['rented'][lead_mask]
        print(f"Selected {len(stage_leads)} leads with actual rented=1 for clustering")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # NEW: Add a diversity check - ensure we have balanced representation of leads
    # For rented stage, make sure we also have some non-rented leads
    if stage == 'rented':
        # Sample some non-rented leads to ensure diversity in clustering
        non_rented_mask = all_labels['applied'].flatten() > 0
        non_rented_mask &= all_labels['rented'].flatten() == 0
        
        if np.sum(non_rented_mask) > 0:
            print(f"Adding {min(len(stage_leads)//5, np.sum(non_rented_mask))} non-rented leads for more diverse clustering")
            
            # Take a sample of non-rented leads (up to 20% of the rented leads)
            sample_size = min(len(stage_leads)//5, np.sum(non_rented_mask))
            non_rented_indices = np.where(non_rented_mask)[0]
            np.random.seed(random_state)
            sampled_indices = np.random.choice(non_rented_indices, size=sample_size, replace=False)
            
            # Combine with rented leads
            stage_leads = np.concatenate([stage_leads, all_lead_ids[sampled_indices]])
            stage_features = np.concatenate([stage_features, all_features[sampled_indices]])
            stage_labels = np.concatenate([stage_labels, all_labels['rented'][sampled_indices]])
            stage_preds = np.concatenate([stage_preds, all_preds['rented'][sampled_indices]])
    
    # Make sure we have enough leads for clustering
    if len(stage_leads) < n_clusters:
        print(f"Warning: Not enough leads ({len(stage_leads)}) for {n_clusters} clusters")
        n_clusters = max(2, len(stage_leads) // 2)
        print(f"Reducing to {n_clusters} clusters")
    
    # Apply dimensionality reduction if needed
    if use_umap and stage_features.shape[1] > 50:
        try:
            import umap
            # NEW: Add more UMAP parameters for better visualization
            reducer = umap.UMAP(
                n_neighbors=30,
                min_dist=0.1,
                metric='euclidean',
                random_state=random_state
            )
            print(f"Reducing dimensionality from {stage_features.shape[1]} features using UMAP...")
            stage_features_reduced = reducer.fit_transform(stage_features)
            
            # Check if reduction was successful
            if stage_features_reduced.shape[1] != 2:
                print(f"Warning: UMAP returned {stage_features_reduced.shape[1]} dimensions instead of 2")
                # Retry with different parameters if needed
                if stage_features_reduced.shape[1] > 2:
                    print("Retrying UMAP with different parameters...")
                    reducer = umap.UMAP(n_components=2, random_state=random_state)
                    stage_features_reduced = reducer.fit_transform(stage_features)
        except ImportError:
            print("UMAP not installed. Using original features.")
            stage_features_reduced = stage_features
        except Exception as e:
            print(f"UMAP error: {str(e)}. Using original features.")
            stage_features_reduced = stage_features
    else:
        # If we have too many features, use PCA instead
        if stage_features.shape[1] > 100:
            try:
                from sklearn.decomposition import PCA
                print(f"Using PCA to reduce {stage_features.shape[1]} features to 50...")
                pca = PCA(n_components=50, random_state=random_state)
                stage_features_reduced = pca.fit_transform(stage_features)
            except Exception as e:
                print(f"PCA error: {str(e)}. Using original features.")
                stage_features_reduced = stage_features
        else:
            stage_features_reduced = stage_features
    
    # NEW: Determine optimal number of clusters
    optimal_n_clusters = n_clusters
    try:
        from sklearn.metrics import silhouette_score
        # Test a range of cluster counts if we have enough samples
        if len(stage_features_reduced) > 3 * n_clusters:
            print("Finding optimal number of clusters...")
            silhouette_scores = []
            cluster_range = range(2, min(15, n_clusters * 2, len(stage_features_reduced) // 10))
            
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(stage_features_reduced)
                score = silhouette_score(stage_features_reduced, cluster_labels)
                silhouette_scores.append(score)
                print(f"  {n} clusters: silhouette score = {score:.4f}")
            
            # Find the best number of clusters
            optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {optimal_n_clusters} (score: {max(silhouette_scores):.4f})")
    except Exception as e:
        print(f"Error finding optimal clusters: {str(e)}. Using original value ({n_clusters}).")
    
    # Apply clustering with optimal cluster count
    n_clusters = optimal_n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    print(f"Clustering {len(stage_features_reduced)} leads into {n_clusters} clusters...")
    cluster_labels = kmeans.fit_predict(stage_features_reduced)
    
    # Count leads per cluster
    cluster_counts = {}
    for i in range(n_clusters):
        cluster_counts[i] = np.sum(cluster_labels == i)
    
    # Analyze clusters for class distribution
    print("\nAnalyzing cluster composition:")
    cluster_conversion_rates = {}
    cluster_stats = {}
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        # Only calculate if we have at least 5 leads in the cluster
        if np.sum(cluster_mask) >= 5:
            # Calculate the conversion rate for this cluster
            conversion_rate = np.mean(stage_labels[cluster_mask])
            cluster_conversion_rates[i] = conversion_rate
            
            # Store other stats
            cluster_stats[i] = {
                'size': np.sum(cluster_mask),
                'conversion_rate': conversion_rate,
                'positive_count': np.sum(stage_labels[cluster_mask]),
                'avg_prediction': np.mean(stage_preds[cluster_mask])
            }
            
            # Print summary
            print(f"Cluster {i}: {np.sum(cluster_mask)} leads, " + 
                  f"conversion: {conversion_rate:.1%}, " +
                  f"avg prediction: {np.mean(stage_preds[cluster_mask]):.4f}")
            
    # Identify suspicious clusters (100% or 0% conversion)
    suspicious_clusters = []
    for i, stats in cluster_stats.items():
        if stats['size'] > 5 and (stats['conversion_rate'] == 1.0 or stats['conversion_rate'] == 0.0):
            suspicious_clusters.append(i)
            print(f"WARNING: Cluster {i} has {stats['conversion_rate']*100:.0f}% conversion rate " +
                  f"({stats['positive_count']}/{stats['size']} leads)")
    
    if suspicious_clusters:
        print("\nDetected potentially suspicious clusters with extreme conversion rates.")
        print("This could indicate:")
        print("1. Perfect feature patterns that predict the outcome")
        print("2. Data leakage - model using target information directly")
        print("3. Overfitting in certain regions of feature space")
        print("Consider checking feature importance and correlation with target.")
    
    # Create visualization based on dimensionality reduction method
    if stage_features_reduced.shape[1] == 2:
        plt.figure(figsize=(12, 10))
        
        # NEW: Better visualization with distinct colors for true labels
        true_status = stage_labels.flatten() > 0.5
        
        # Plot clusters with different shapes
        plt.scatter(stage_features_reduced[~true_status, 0], 
                   stage_features_reduced[~true_status, 1],
                   c=cluster_labels[~true_status], 
                   marker='o',
                   alpha=0.6, 
                   cmap='tab20')
        
        plt.scatter(stage_features_reduced[true_status, 0], 
                   stage_features_reduced[true_status, 1],
                   c=cluster_labels[true_status], 
                   marker='^',  # Use triangles for positive examples
                   alpha=0.7, 
                   cmap='tab20')
        
        # Mark centroids
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   marker='x', s=150, linewidths=3,
                   color='red', zorder=10)
        
        # Add annotations for conversion rates
        for i, (x, y) in enumerate(centroids):
            if i in cluster_conversion_rates:
                plt.annotate(f"{i}: {cluster_conversion_rates[i]:.1%}", 
                             (x, y), 
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=12,
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                             zorder=11)
        
        plt.colorbar(label='Cluster')
        plt.title(f'{stage.capitalize()} Stage Clustering\n' + 
                 f'(^: {stage} positive, o: {stage} negative)')
        plt.tight_layout()
        
        # Save as bytes
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plot_bytes = buf.getvalue()
        plt.close()
    else:
        plot_bytes = None
    
    results = {
        'stage': stage,
        'n_clusters': n_clusters,
        'lead_ids': stage_leads,
        'cluster_labels': cluster_labels,
        'centroids': kmeans.cluster_centers_,
        'cluster_counts': cluster_counts,
        'cluster_stats': cluster_stats, 
        'suspicious_clusters': suspicious_clusters,
        'plot_bytes': plot_bytes
    }
    
    return results


def evaluate_cluster_specific_metrics(model, dataloader, cluster_assignments, device='cuda'):
    """
    Calculate metrics for each cluster to identify where the model performs well or poorly.
    This helps pinpoint specific lead segments that need improvement.
    
    Args:
        model: Trained model
        dataloader: DataLoader with validation/test data
        cluster_assignments: Dictionary mapping lead_ids to cluster labels
        device: Device to use for prediction
        
    Returns:
        Dictionary with metrics per cluster
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    
    model.eval()
    
    # Collect predictions and labels per cluster
    cluster_data = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating cluster metrics"):
            cat_in = batch[0].to(device).long()  # Ensure long type for embedding
            num_in = batch[1].to(device).float()  # Ensure float type for numerical
            toured_labels = batch[2].cpu().numpy()
            applied_labels = batch[3].cpu().numpy()
            rented_labels = batch[4].cpu().numpy()
            lead_ids = batch[5].cpu().numpy() if len(batch) > 5 else np.arange(len(cat_in))
            
            # Forward pass
            outputs = model(cat_in, num_in)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                toured_pred, applied_pred, rented_pred = outputs[:3]
            else:
                toured_pred, applied_pred, rented_pred = outputs
            
            # Convert to numpy
            toured_pred = toured_pred.cpu().numpy()
            applied_pred = applied_pred.cpu().numpy()
            rented_pred = rented_pred.cpu().numpy()
            
            # Find cluster for each lead
            for i, lead_id in enumerate(lead_ids):
                if lead_id in cluster_assignments:
                    cluster = cluster_assignments[lead_id]
                    
                    # Initialize cluster data structure
                    if cluster not in cluster_data:
                        cluster_data[cluster] = {
                            'toured_preds': [], 'toured_labels': [],
                            'applied_preds': [], 'applied_labels': [],
                            'rented_preds': [], 'rented_labels': [],
                            'lead_ids': []
                        }
                    
                    # Store predictions and labels
                    cluster_data[cluster]['toured_preds'].append(toured_pred[i])
                    cluster_data[cluster]['toured_labels'].append(toured_labels[i])
                    cluster_data[cluster]['applied_preds'].append(applied_pred[i])
                    cluster_data[cluster]['applied_labels'].append(applied_labels[i])
                    cluster_data[cluster]['rented_preds'].append(rented_pred[i])
                    cluster_data[cluster]['rented_labels'].append(rented_labels[i])
                    cluster_data[cluster]['lead_ids'].append(lead_id)
    
    # Calculate metrics per cluster
    cluster_metrics = {}
    stages = ['toured', 'applied', 'rented']
    
    print("\nCluster-specific metrics:")
    print("-" * 60)
    print(f"{'Cluster':<8} {'Size':<6} {'Stage':<10} {'AUC':<8} {'APR':<8} {'P@10':<8}")
    print("-" * 60)
    
    for cluster, data in sorted(cluster_data.items()):
        cluster_metrics[cluster] = {}
        
        for stage in stages:
            preds = np.array(data[f'{stage}_preds']).flatten()
            labels = np.array(data[f'{stage}_labels']).flatten()
            
            # Skip clusters with too few samples or single class
            if len(preds) < 10 or len(np.unique(labels)) < 2:
                cluster_metrics[cluster][stage] = {
                    'auc': np.nan, 'apr': np.nan, 'p@10': np.nan
                }
                continue
            
            # Calculate metrics
            metrics = {
                'auc': roc_auc_score(labels, preds),
                'apr': average_precision_score(labels, preds)
            }
            
            # Calculate precision@10
            k = min(10, len(preds))
            top_indices = np.argsort(preds)[-k:]
            metrics['p@10'] = np.mean(labels[top_indices])
            
            cluster_metrics[cluster][stage] = metrics
            
            # Print metrics
            print(f"{cluster:<8} {len(preds):<6} {stage:<10} "
                  f"{metrics['auc']:<8.4f} {metrics['apr']:<8.4f} {metrics['p@10']:<8.4f}")
    
    # Aggregate summary: identify problematic segments
    problem_clusters = {}
    for stage in stages:
        # Find clusters with below-average AUC
        stage_aucs = [metrics[stage]['auc'] for cluster, metrics in cluster_metrics.items() 
                     if not np.isnan(metrics[stage]['auc'])]
        
        if not stage_aucs:
            continue
            
        avg_auc = np.mean(stage_aucs)
        
        # Identify poor performers
        poor_clusters = [
            (cluster, metrics[stage]['auc']) 
            for cluster, metrics in cluster_metrics.items()
            if not np.isnan(metrics[stage]['auc']) and metrics[stage]['auc'] < avg_auc * 0.9
        ]
        
        if poor_clusters:
            problem_clusters[stage] = sorted(poor_clusters, key=lambda x: x[1])
    
    # Print problem clusters
    if problem_clusters:
        print("\nProblem clusters requiring attention:")
        for stage, clusters in problem_clusters.items():
            print(f"\n{stage.capitalize()} stage:")
            for cluster, auc in clusters:
                size = len(cluster_data[cluster][f'{stage}_preds'])
                print(f"  Cluster {cluster}: AUC={auc:.4f}, Size={size} leads")
    
    return {
        'cluster_metrics': cluster_metrics,
        'problem_clusters': problem_clusters
    }