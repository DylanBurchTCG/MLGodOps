import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import os
import time
from contextlib import nullcontext


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
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


def precision_at_k(y_true, y_scores, k):
    """Calculate precision@k for binary classification"""
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_true = y_true[top_k_indices]
    return np.mean(top_k_true)


def train_tour_model(
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler=None,
        num_epochs=50,
        device='cuda',
        focal_alpha=0.25,
        focal_gamma=2.0,
        model_save_path='tour_model.pt',
        early_stopping_patience=5,
        mixed_precision=True,
        max_grad_norm=1.0
):
    """
    Simplified training function for tour prediction model
    """
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

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0

        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(progress_bar):
            cat_features = batch[0].to(device)
            num_features = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():
                preds, _ = model(cat_features, num_features)

                # Ensure dimensions match
                if preds.dim() != labels.dim():
                    if preds.dim() == 2 and labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    elif preds.dim() == 1 and labels.dim() == 2:
                        preds = preds.unsqueeze(1)

                # Calculate loss
                loss = focal_loss_fn(preds, labels)

            # Backward pass with gradient scaling
            optimizer.zero_grad()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        if scheduler is not None:
            scheduler.step()

        train_loss /= max(1, train_steps)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]"):
                cat_features = batch[0].to(device)
                num_features = batch[1].to(device)
                labels = batch[2].to(device)

                # Forward pass
                preds, _ = model(cat_features, num_features)

                # Ensure dimensions match
                if preds.dim() != labels.dim():
                    if preds.dim() == 2 and labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    elif preds.dim() == 1 and labels.dim() == 2:
                        preds = preds.unsqueeze(1)

                # Calculate loss
                loss = focal_loss_fn(preds, labels)
                val_loss += loss.item()

                # Store predictions and labels
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= len(valid_loader)
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
            print(f"✓ New best AUC: {auc:.4f}, saving model")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {auc:.4f}, Val APR: {apr:.4f}")
        print(f"Precision@k: {', '.join([f'P@{k}={v:.4f}' for k, v in p_at_k.items()])}")

    print("\nTraining completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {model_save_path}")

    # Load best model
    model.load_state_dict(torch.load(model_save_path))

    return model, history