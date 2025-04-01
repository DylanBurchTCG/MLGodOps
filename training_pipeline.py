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


# ----------------
# Dataset Class
# ----------------
class LeadDataset(Dataset):
    def __init__(self, categorical_features, numerical_features,
                 toured_labels=None, applied_labels=None, rented_labels=None,
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
                           device='cuda'):
    """
    Ranks leads in a funnel:
     1) Sort by Toured prob => top K
     2) Among them, sort by Applied prob => top K
     3) Among them, sort by Rented prob => top K
    Returns a dict with selected IDs and excluded IDs for each stage.
    """
    model.eval()

    all_lead_ids = []
    all_toured_preds = []
    all_applied_preds = []
    all_rented_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ranking leads"):
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
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

    # Ensure k values don't exceed the dataset size
    toured_k = min(toured_k, len(all_lead_ids))

    # Stage 1: Toured
    toured_indices = np.argsort(all_toured_preds)[::-1][:toured_k]
    toured_selected_ids = all_lead_ids[toured_indices]
    toured_scores = all_toured_preds[toured_indices]
    toured_excluded_ids = np.setdiff1d(all_lead_ids, toured_selected_ids)

    print(f"Selected {len(toured_selected_ids)} out of {len(all_lead_ids)} leads for touring")

    # Stage 2: Among the top Toured, rank by Applied
    applied_k = min(applied_k, len(toured_selected_ids))
    applied_subset = all_applied_preds[toured_indices]
    applied_indices = np.argsort(applied_subset)[::-1][:applied_k]
    applied_selected_ids = toured_selected_ids[applied_indices]
    applied_scores = applied_subset[applied_indices]
    applied_excluded_ids = np.setdiff1d(toured_selected_ids, applied_selected_ids)

    print(f"Selected {len(applied_selected_ids)} out of {len(toured_selected_ids)} leads for applications")

    # Stage 3: Among top Applied, rank by Rented
    rented_k = min(rented_k, len(applied_selected_ids))
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
    toured_criterion = nn.BCELoss()
    applied_criterion = nn.BCELoss()
    rented_criterion = nn.BCELoss()

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
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
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
        'toured_p50': [], 'applied_p50': [], 'rented_p50': []  # Added Precision@50
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
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
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
                            toured_loss = torch.tensor(0.5, device=device)
                            print(f"Warning: Replaced extreme toured loss")

                        if torch.isnan(applied_loss) or torch.isinf(applied_loss) or applied_loss > 10:
                            applied_loss = torch.tensor(0.5, device=device)
                            print(f"Warning: Replaced extreme applied loss")

                        if torch.isnan(rented_loss) or torch.isinf(rented_loss) or rented_loss > 10:
                            rented_loss = torch.tensor(0.5, device=device)
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

                        # Final sanity check with strict limits
                        if not torch.isfinite(loss) or loss > 10:
                            print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                            loss = torch.tensor(1.0, device=device)  # Safe fallback

                        # gradient accumulation
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stricter gradient clipping
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # No mixed precision - similar logic but without scaler
                    outputs = model(categorical_inputs, numerical_inputs, lead_ids)
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        toured_pred, applied_pred, rented_pred = outputs[:3]
                    else:
                        toured_pred, applied_pred, rented_pred = outputs

                    # Add epsilon to prevent exact 0s and 1s
                    epsilon = 1e-7
                    toured_pred = torch.clamp(toured_pred, epsilon, 1 - epsilon)
                    applied_pred = torch.clamp(applied_pred, epsilon, 1 - epsilon)
                    rented_pred = torch.clamp(rented_pred, epsilon, 1 - epsilon)

                    # Standard loss calculation
                    toured_loss = toured_criterion(toured_pred, toured_labels)
                    applied_loss = applied_criterion(applied_pred, applied_labels)
                    rented_loss = rented_criterion(rented_pred, rented_labels)

                    # Check for extreme values
                    if torch.isnan(toured_loss) or torch.isinf(toured_loss) or toured_loss > 10:
                        toured_loss = torch.tensor(0.5, device=device)
                        print(f"Warning: Replaced extreme toured loss")

                    if torch.isnan(applied_loss) or torch.isinf(applied_loss) or applied_loss > 10:
                        applied_loss = torch.tensor(0.5, device=device)
                        print(f"Warning: Replaced extreme applied loss")

                    if torch.isnan(rented_loss) or torch.isinf(rented_loss) or rented_loss > 10:
                        rented_loss = torch.tensor(0.5, device=device)
                        print(f"Warning: Replaced extreme rented loss")

                    # Standard loss calculation
                    bce_loss = toured_weight * toured_loss + \
                               applied_weight * applied_loss + \
                               rented_weight * rented_loss

                    # Add ranking loss if model supports it and batch size large enough
                    if ranking_criterion and categorical_inputs.size(0) >= 10:
                        try:
                            from model_architecture import ListMLELoss
                            ranking_module = ListMLELoss()

                            toured_ranking_loss = ranking_module(toured_pred.squeeze(), toured_labels.squeeze())
                            applied_ranking_loss = ranking_module(applied_pred.squeeze(), applied_labels.squeeze())
                            rented_ranking_loss = ranking_module(rented_pred.squeeze(), rented_labels.squeeze())

                            # Clamp ranking losses with stricter bounds
                            toured_ranking_loss = torch.clamp(toured_ranking_loss, -5, 5)
                            applied_ranking_loss = torch.clamp(applied_ranking_loss, -5, 5)
                            rented_ranking_loss = torch.clamp(rented_ranking_loss, -5, 5)

                            rank_loss = toured_weight * toured_ranking_loss + \
                                        applied_weight * applied_ranking_loss + \
                                        rented_weight * rented_ranking_loss

                            # Combine BCE and ranking loss
                            loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * rank_loss
                        except Exception as e:
                            print(f"Ranking loss failed: {str(e)}, using BCE only")
                            loss = bce_loss
                    else:
                        loss = bce_loss

                    # Final sanity check
                    if not torch.isfinite(loss) or loss > 10:
                        print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                        loss = torch.tensor(1.0, device=device)  # Safe fallback

                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Stricter gradient clipping
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item() * gradient_accumulation_steps
                train_iter.set_postfix({'loss': float(loss.item() * gradient_accumulation_steps)})
                num_batches += 1

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                print(f"Batch shapes: cat={categorical_inputs.shape}, num={numerical_inputs.shape}")
                # Skip this batch and continue
                continue

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
            categorical_inputs = batch[0].to(device)
            numerical_inputs = batch[1].to(device)
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
                        toured_loss = torch.tensor(1.0, device=device)
                    if not torch.isfinite(applied_loss) or applied_loss > 50:
                        applied_loss = torch.tensor(1.0, device=device)
                    if not torch.isfinite(rented_loss) or rented_loss > 50:
                        rented_loss = torch.tensor(1.0, device=device)

                    batch_loss = (toured_weight * toured_loss +
                                  applied_weight * applied_loss +
                                  rented_weight * rented_loss)

                    # Stricter clipping for batch loss
                    if not torch.isfinite(batch_loss) or batch_loss > 50:
                        batch_loss = torch.tensor(3.0, device=device)

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

        # Print metrics with clear formatting and line breaks
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("\nMetrics:")
        print(f"{'Stage':<10} {'AUC':<10} {'APR':<10} {'P@10':<10} {'P@50':<10}")
        print("-" * 80)
        print(f"{'Toured':<10} {toured_auc:.4f}     {toured_apr:.4f}     {toured_p10:.4f}     {toured_p50:.4f}")
        print(f"{'Applied':<10} {applied_auc:.4f}     {applied_apr:.4f}     {applied_p10:.4f}     {applied_p50:.4f}")
        print(f"{'Rented':<10} {rented_auc:.4f}     {rented_apr:.4f}     {rented_p10:.4f}     {rented_p50:.4f}")
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