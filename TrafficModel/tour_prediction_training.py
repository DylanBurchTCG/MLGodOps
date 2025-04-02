import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from contextlib import nullcontext

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.dim() != targets.dim():
            if inputs.dim() == 1 and targets.dim() == 2:
                inputs = inputs.unsqueeze(1)  # Make inputs 2D
            elif inputs.dim() == 2 and targets.dim() == 1:
                targets = targets.unsqueeze(1)  # Make targets 2D
        
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class TourDataset(Dataset):
    def __init__(self, categorical_features, numerical_features, toured_labels, lead_ids=None):
        self.categorical_features = categorical_features.long() if categorical_features is not None else None
        self.numerical_features = numerical_features.float() if numerical_features is not None else None
        self.toured_labels = toured_labels.float() if toured_labels is not None else None
        self.lead_ids = lead_ids
    
    def __len__(self):
        return len(self.toured_labels)
    
    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([], dtype=torch.long),
            self.numerical_features[idx] if self.numerical_features is not None else torch.tensor([], dtype=torch.float),
            self.toured_labels[idx]
        ]
        
        if self.lead_ids is not None:
            items.append(self.lead_ids[idx].clone().detach())
            
        return tuple(items)

def precision_at_k(y_true, y_scores, k):
    # Sort by score in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    # Take top k predictions
    top_k_indices = sorted_indices[:k]
    # Get corresponding true labels
    top_k_true = y_true[top_k_indices]
    # Calculate precision
    return np.mean(top_k_true)

def train_tour_prediction_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device='cuda',
    focal_alpha=0.25,  # Changed from 0.3712 to default value
    focal_gamma=2.0,   # Changed from 1.5 to default value
    class_weights=None,
    model_save_path='tour_model.pt',
    grad_accumulation_steps=1,
    early_stopping_patience=5,
    mixed_precision=True,
    max_grad_norm=1.0  # Added gradient clipping
):
    # Initialize focal loss
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    # Enable mixed precision if requested
    use_amp = mixed_precision and device == 'cuda' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_apr': [],
        'val_p_at_k': {10: [], 50: [], 100: []}
    }
    
    # Initialize early stopping
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # Training loop
        for batch_idx, (cat_features, num_features, labels, *_) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')):
            cat_features = cat_features.to(device)
            num_features = num_features.to(device)
            labels = labels.to(device)
            
            # Print shapes for first batch in first epoch for debugging
            if batch_idx == 0 and epoch == 0:
                print(f"Training shapes - Cat: {cat_features.shape}, Num: {num_features.shape}, Labels: {labels.shape}")
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda') if use_amp else nullcontext():
                preds, _ = model(cat_features, num_features)
                # Ensure preds and labels have same dimensions
                if preds.dim() != labels.dim():
                    if preds.dim() == 2 and labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    elif preds.dim() == 1 and labels.dim() == 2:
                        preds = preds.unsqueeze(1)
                loss = focal_loss_fn(preds, labels)
                loss = loss / grad_accumulation_steps
            
            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                if (batch_idx + 1) % grad_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            
            train_loss += loss.item() * grad_accumulation_steps
            train_steps += 1
            
            if scheduler is not None:
                scheduler.step()
        
        train_loss /= max(1, train_steps)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_iter):
                categorical_inputs = batch[0].to(device) if batch[0].size(1) > 0 else None
                numerical_inputs = batch[1].to(device) if batch[1].size(1) > 0 else None
                toured_labels = batch[2].to(device)
                
                # Handle empty inputs
                if (categorical_inputs is None or categorical_inputs.size(0) == 0) and \
                   (numerical_inputs is None or numerical_inputs.size(0) == 0):
                    continue
                    
                try:
                    # Forward pass
                    toured_pred, _ = model(categorical_inputs, numerical_inputs)
                    
                    # Print shapes for debugging (first batch only)
                    if batch_idx == 0 and epoch == 0:
                        print(f"Validation shapes - Pred: {toured_pred.shape}, Labels: {toured_labels.shape}")
                    
                    # Ensure same dimensions
                    if toured_pred.dim() != toured_labels.dim():
                        if toured_pred.dim() == 2 and toured_labels.dim() == 1:
                            toured_labels = toured_labels.unsqueeze(1)
                        elif toured_pred.dim() == 1 and toured_labels.dim() == 2:
                            toured_pred = toured_pred.unsqueeze(1)
                    
                    # Use same loss calculation as training
                    focal = focal_loss_fn(toured_pred, toured_labels)
                    bce = F.binary_cross_entropy(toured_pred, toured_labels)
                    loss = 0.7 * focal + 0.3 * bce
                    
                    val_loss += loss.item()
                    
                    # Store predictions and labels
                    val_preds.append(toured_pred.cpu().numpy())
                    val_labels.append(toured_labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    print(f"Shapes - Pred: {toured_pred.shape if 'toured_pred' in locals() else 'N/A'}, Labels: {toured_labels.shape}")
                    continue
        
        val_loss /= max(1, len(valid_loader))
        history['val_loss'].append(val_loss)
        
        # Flatten predictions and labels
        val_preds = np.concatenate(val_preds).flatten()
        val_labels = np.concatenate(val_labels).flatten()
        
        # Calculate metrics
        auc = roc_auc_score(val_labels, val_preds)
        apr = average_precision_score(val_labels, val_preds)
        
        # Precision@k metrics
        k_values = [10, 50, 100]
        p_at_k = {}
        for k in k_values:
            if len(val_labels) >= k:
                p_at_k[k] = precision_at_k(val_labels, val_preds, k)
                history['val_p_at_k'][k].append(p_at_k[k])
            else:
                p_at_k[k] = np.nan
        
        # Update history
        history['val_auc'].append(auc)
        history['val_apr'].append(apr)
        
        # Early stopping check
        if auc > best_val_auc:
            best_val_auc = auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {auc:.4f}")
        print(f"Val APR: {apr:.4f}")
        print(f"Precision@k: {p_at_k}")
        
    return model, history
