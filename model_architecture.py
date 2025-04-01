import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class EmbeddingLayer(nn.Module):
    """Handles embedding of categorical features and processing of numerical features"""

    def __init__(self, categorical_dims, embedding_dims, numerical_dim):
        super().__init__()
        self.categorical_layers = nn.ModuleList([
            nn.Linear(1, emb_dim) for cat_dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        self.numerical_bn = nn.BatchNorm1d(numerical_dim) if numerical_dim > 0 else None

    def forward(self, categorical_inputs, numerical_inputs):
        embedded_features = []
        if len(self.categorical_layers) > 0:
            for i, layer in enumerate(self.categorical_layers):
                cat_feature = categorical_inputs[:, i].view(-1, 1)
                embedded_features.append(layer(cat_feature))

        if self.numerical_bn is not None:
            numerical_features = self.numerical_bn(numerical_inputs)
        else:
            numerical_features = numerical_inputs

        all_features = []
        if embedded_features:
            cat_out = torch.cat(embedded_features, dim=1)
            all_features.append(cat_out)
        if numerical_features.size(1) > 0:
            all_features.append(numerical_features)

        if all_features:
            return torch.cat(all_features, dim=1)
        else:
            return torch.zeros((categorical_inputs.size(0), 0), device=categorical_inputs.device)


class TabularTransformerBlock(nn.Module):
    """Transformer block adapted for tabular data"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Reshape for attention
        batch_size = x.size(0)

        # Ensure x is shaped appropriately for the transformer block
        if x.dim() == 2:
            # Reshape to (batch_size, seq_len=1, embed_dim)
            if x.size(1) == self.embed_dim:
                x = x.unsqueeze(1)  # Add sequence dimension
            else:
                # Handle the case where input dimension doesn't match embed_dim
                # Reshape to have seq_len such that seq_len * embed_dim = x.size(1)
                seq_len = max(1, x.size(1) // self.embed_dim)
                # Pad if needed
                padded_dim = seq_len * self.embed_dim
                if padded_dim > x.size(1):
                    pad = torch.zeros(batch_size, padded_dim - x.size(1), device=x.device)
                    x = torch.cat([x, pad], dim=1)
                # Reshape to (batch_size, seq_len, embed_dim)
                x = x.view(batch_size, seq_len, self.embed_dim)

        # Apply attention mechanism
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        # Apply feed-forward
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        out = self.norm2(x + ffn_out)

        # Use mean pooling to get fixed-size representation
        out = out.mean(dim=1)  # Shape: (batch_size, embed_dim)
        return out


class PredictionHead(nn.Module):
    """Prediction head for binary classification"""

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ListMLELoss(nn.Module):
    """
    ListMLE (Maximum Likelihood Estimation) loss for learning to rank.

    Paper: "Listwise Approach to Learning to Rank - Theory and Algorithm"
    """

    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, scores, targets):
        """
        Args:
            scores: predicted scores, shape [batch_size, 1]
            targets: ground truth labels, shape [batch_size, 1]
        """
        # Reshape if needed
        if scores.dim() > 1:
            scores = scores.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Sort targets in descending order
        sorted_targets, indices = torch.sort(targets, descending=True, dim=0)

        # Reorder scores according to target sorting
        ordered_scores = torch.gather(scores, 0, indices)

        # Apply softmax to get probabilities
        scores_softmax = F.log_softmax(ordered_scores, dim=0)

        # Compute loss
        loss = -torch.sum(scores_softmax)

        return loss


class MultiTaskCascadedLeadFunnelModel(nn.Module):
    """
    Enhanced multi-stage cascade model with shared representations:
    1. Process all leads through initial embedding and shared transformer blocks
    2. Then branch out for stage-specific predictions with additional task-specific layers
    3. Maintains the cascaded selection process between stages
    """

    def __init__(self,
                 categorical_dims,
                 embedding_dims,
                 numerical_dim,
                 transformer_dim,
                 num_heads,
                 ff_dim,
                 head_hidden_dims,
                 dropout=0.2,
                 toured_k=2000,
                 applied_k=1000,
                 rented_k=250):  # Updated default to 250 for rented_k
        super().__init__()

        # Stage filtering parameters
        self.toured_k = toured_k
        self.applied_k = applied_k
        self.rented_k = rented_k

        # 1. Embedding layer
        self.embedding_layer = EmbeddingLayer(categorical_dims, embedding_dims, numerical_dim)

        # 2. Projection to transformer dimension
        total_emb_dim = sum(embedding_dims) + numerical_dim
        self.projection = nn.Linear(total_emb_dim, transformer_dim)

        # 3. Shared transformer blocks for common representation
        self.shared_transformer = nn.Sequential(
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout),
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)
        )

        # 4. Task-specific transformer blocks
        self.transformer_toured = TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)
        self.transformer_applied = TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)
        self.transformer_rented = TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)

        # 5. Prediction heads for each stage
        self.toured_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.applied_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.rented_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)

    def forward(self, categorical_inputs, numerical_inputs, lead_ids=None, is_training=True):
        """
        Forward pass with cascading stages and filtering between them.
        During training: process all leads through all stages but with gradient tracking
        During inference: filter at each stage before proceeding
        """
        batch_size = categorical_inputs.size(0)
        device = categorical_inputs.device

        # Initial embedding for all leads
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)

        # Shared representation
        shared_features = self.shared_transformer(x)

        # Stage 1: Toured prediction
        toured_features = self.transformer_toured(shared_features)
        toured_pred = self.toured_head(toured_features)

        # Handle empty batches gracefully
        if batch_size == 0:
            return (toured_pred,
                    torch.zeros((0, 1), device=device),
                    torch.zeros((0, 1), device=device),
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device))

        # Ensure toured_pred is properly shaped
        if toured_pred.dim() == 1:
            toured_pred = toured_pred.unsqueeze(1)

        # Select top toured leads - use min to avoid errors with small batches
        k_toured = min(self.toured_k, batch_size)
        if toured_pred.squeeze().dim() == 0:  # Handle scalar case
            toured_indices = torch.tensor([0], device=device)
        else:
            _, toured_indices = torch.topk(toured_pred.squeeze(), k_toured)

        if is_training:
            # During training, don't actually filter but track selection for loss weighting
            applied_features = self.transformer_applied(shared_features)
            applied_pred = self.applied_head(applied_features)

            # Ensure applied_pred is properly shaped
            if applied_pred.dim() == 1:
                applied_pred = applied_pred.unsqueeze(1)

            # Select top applied leads from toured subset - handle empty toured_indices
            if len(toured_indices) == 0:
                applied_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Handle scalar case
                if applied_pred.squeeze()[toured_indices].dim() == 0:
                    applied_indices_local = torch.tensor([0], device=device)
                else:
                    k_applied = min(self.applied_k, len(toured_indices))
                    _, applied_indices_local = torch.topk(applied_pred.squeeze()[toured_indices], k_applied)
                applied_indices = toured_indices[applied_indices_local]

            # Process all leads for the rented stage
            rented_features = self.transformer_rented(shared_features)
            rented_pred = self.rented_head(rented_features)

            # Ensure rented_pred is properly shaped
            if rented_pred.dim() == 1:
                rented_pred = rented_pred.unsqueeze(1)

            # Final ranking - handle empty applied_indices
            if len(applied_indices) == 0:
                rented_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Handle scalar case
                if rented_pred.squeeze()[applied_indices].dim() == 0:
                    rented_indices_local = torch.tensor([0], device=device)
                else:
                    k_rented = min(self.rented_k, len(applied_indices))
                    _, rented_indices_local = torch.topk(rented_pred.squeeze()[applied_indices], k_rented)
                rented_indices = applied_indices[rented_indices_local]

        else:
            # During inference, actually filter leads between stages
            # Handle empty toured_indices
            if len(toured_indices) == 0:
                # Return empty results if no leads selected
                return (toured_pred,
                        torch.zeros((batch_size, 1), device=device),
                        torch.zeros((batch_size, 1), device=device),
                        toured_indices,
                        torch.tensor([], dtype=torch.long, device=device),
                        torch.tensor([], dtype=torch.long, device=device))

            # Only process toured selected leads
            applied_features = self.transformer_applied(shared_features[toured_indices])
            applied_pred_subset = self.applied_head(applied_features)

            # Expand predictions back to full batch size for output consistency
            applied_pred = torch.zeros((batch_size, 1), device=device)
            applied_pred[toured_indices] = applied_pred_subset

            # Select top applied leads - handle empty or singleton toured_indices
            if len(toured_indices) == 0:
                applied_indices = torch.tensor([], dtype=torch.long, device=device)
                applied_indices_local = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Handle scalar case
                if applied_pred_subset.squeeze().dim() == 0:
                    applied_indices_local = torch.tensor([0], device=device)
                else:
                    k_applied = min(self.applied_k, len(toured_indices))
                    _, applied_indices_local = torch.topk(applied_pred_subset.squeeze(), k_applied)
                applied_indices = toured_indices[applied_indices_local]

            # Handle empty applied_indices
            if len(applied_indices) == 0:
                # Return with empty rented predictions
                rented_pred = torch.zeros((batch_size, 1), device=device)
                return (toured_pred, applied_pred, rented_pred,
                        toured_indices, applied_indices,
                        torch.tensor([], dtype=torch.long, device=device))

            # Only process applied selected leads
            rented_features = self.transformer_rented(shared_features[applied_indices])
            rented_pred_subset = self.rented_head(rented_features)

            # Expand predictions back to full batch size
            rented_pred = torch.zeros((batch_size, 1), device=device)
            rented_pred[applied_indices] = rented_pred_subset

            # Select top rented leads
            if len(applied_indices) == 0:
                rented_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Handle scalar case
                if rented_pred_subset.squeeze().dim() == 0:
                    rented_indices_local = torch.tensor([0], device=device)
                else:
                    k_rented = min(self.rented_k, len(applied_indices))
                    _, rented_indices_local = torch.topk(rented_pred_subset.squeeze(), k_rented)
                rented_indices = applied_indices[rented_indices_local]

        return toured_pred, applied_pred, rented_pred, toured_indices, applied_indices, rented_indices


# Keep this for backward compatibility
CascadedLeadFunnelModel = MultiTaskCascadedLeadFunnelModel


def precision_at_k(y_true, y_scores, k):
    """
    Calculate precision@k for binary classification

    Args:
        y_true: Binary ground truth, shape [n_samples]
        y_scores: Predicted scores, shape [n_samples]
        k: Number of top items to consider

    Returns:
        Precision@k score
    """
    # Get indices of top-k scores
    top_indices = np.argsort(y_scores)[-k:]

    # Count true positives among top-k predictions
    num_relevant = np.sum(y_true[top_indices])

    # Calculate precision@k
    return num_relevant / k if k > 0 else 0.0


# Training function for cascaded model with ranking loss
def train_cascaded_model(model,
                         train_loader,
                         valid_loader,
                         optimizer,
                         scheduler,
                         num_epochs=30,
                         toured_weight=1.0,
                         applied_weight=1.0,
                         rented_weight=2.0,
                         ranking_weight=0.3,  # Weight for the ranking loss component
                         device='cuda',
                         early_stopping_patience=5,
                         model_save_path='best_model.pt',
                         mixed_precision=True,
                         gradient_accumulation_steps=1,
                         verbose=True):
    """
    Training loop for cascaded model with ranking loss
    """
    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device == 'cuda') else None

    # Loss functions
    # Standard classification losses
    toured_criterion = nn.BCELoss(reduction='none')
    applied_criterion = nn.BCELoss(reduction='none')
    rented_criterion = nn.BCELoss(reduction='none')

    # Ranking loss
    ranking_criterion = ListMLELoss()

    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # -------- TRAINING --------
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        # Use tqdm only if verbose mode is on
        if verbose:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        else:
            train_iter = train_loader

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
                        toured_pred, applied_pred, rented_pred, toured_idx, applied_idx, rented_idx = model(
                            categorical_inputs, numerical_inputs, lead_ids, is_training=True
                        )

                        # Calculate BCE losses with focus on selected indices
                        toured_losses = toured_criterion(toured_pred, toured_labels)

                        # Add loss clipping to prevent extremely large loss values
                        toured_losses = torch.clamp(toured_losses, 0, 100)
                        toured_loss = toured_losses.mean()

                        # For applied, weight higher for toured selected
                        applied_losses = applied_criterion(applied_pred, applied_labels)
                        applied_losses = torch.clamp(applied_losses, 0, 100)

                        toured_mask = torch.zeros_like(applied_losses, device=device)
                        if len(toured_idx) > 0:  # Only set weights if we have selected indices
                            toured_mask[toured_idx] = 2.0  # Higher weight for selected leads
                            # Set mask for non-selected items
                            non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                            toured_mask[non_selected_mask] = 0.5
                        applied_loss = (applied_losses * (toured_mask + 0.1)).mean()

                        # For rented, weight higher for applied selected
                        rented_losses = rented_criterion(rented_pred, rented_labels)
                        rented_losses = torch.clamp(rented_losses, 0, 100)

                        applied_mask = torch.zeros_like(rented_losses, device=device)
                        if len(applied_idx) > 0:  # Only set weights if we have selected indices
                            applied_mask[applied_idx] = 2.0
                            # Set mask for non-selected items
                            non_selected_mask = ~torch.isin(torch.arange(len(applied_mask), device=device), applied_idx)
                            applied_mask[non_selected_mask] = 0.5
                        rented_loss = (rented_losses * (applied_mask + 0.1)).mean()

                        # Standard BCE loss
                        bce_loss = toured_weight * toured_loss + \
                                   applied_weight * applied_loss + \
                                   rented_weight * rented_loss

                        # Add Ranking Loss - try to use it when we have enough batch size for meaningful ranking
                        if categorical_inputs.size(0) >= 10:
                            try:
                                toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                                applied_ranking_loss = ranking_criterion(applied_pred.squeeze(),
                                                                         applied_labels.squeeze())
                                rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                                # Clamp ranking losses too
                                toured_ranking_loss = torch.clamp(toured_ranking_loss, -100, 100)
                                applied_ranking_loss = torch.clamp(applied_ranking_loss, -100, 100)
                                rented_ranking_loss = torch.clamp(rented_ranking_loss, -100, 100)

                                ranking_loss = toured_weight * toured_ranking_loss + \
                                               applied_weight * applied_ranking_loss + \
                                               rented_weight * rented_ranking_loss

                                # Combine BCE and ranking loss
                                loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                            except Exception as e:
                                # Fallback if ranking loss fails
                                print(f"Ranking loss failed: {str(e)}, using BCE only")
                                loss = bce_loss
                        else:
                            # Use only BCE if batch is too small
                            loss = bce_loss

                        # Final sanity check on loss value
                        if not torch.isfinite(loss):
                            print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                            loss = torch.tensor(10.0, device=device)  # Safe fallback

                        # gradient accumulation
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        # Increased gradient clipping threshold to prevent numerical issues
                        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # No mixed precision - implement similar logic
                    toured_pred, applied_pred, rented_pred, toured_idx, applied_idx, rented_idx = model(
                        categorical_inputs, numerical_inputs, lead_ids, is_training=True
                    )

                    # Calculate BCE losses
                    toured_losses = toured_criterion(toured_pred, toured_labels)
                    toured_losses = torch.clamp(toured_losses, 0, 100)
                    toured_loss = toured_losses.mean()

                    applied_losses = applied_criterion(applied_pred, applied_labels)
                    applied_losses = torch.clamp(applied_losses, 0, 100)

                    toured_mask = torch.zeros_like(applied_losses, device=device)
                    if len(toured_idx) > 0:
                        toured_mask[toured_idx] = 2.0
                        non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                        toured_mask[non_selected_mask] = 0.5
                    applied_loss = (applied_losses * (toured_mask + 0.1)).mean()

                    rented_losses = rented_criterion(rented_pred, rented_labels)
                    rented_losses = torch.clamp(rented_losses, 0, 100)

                    applied_mask = torch.zeros_like(rented_losses, device=device)
                    if len(applied_idx) > 0:
                        applied_mask[applied_idx] = 2.0
                        non_selected_mask = ~torch.isin(torch.arange(len(applied_mask), device=device), applied_idx)
                        applied_mask[non_selected_mask] = 0.5
                    rented_loss = (rented_losses * (applied_mask + 0.1)).mean()

                    # Standard BCE loss
                    bce_loss = toured_weight * toured_loss + \
                               applied_weight * applied_loss + \
                               rented_weight * rented_loss

                    # Add Ranking Loss
                    if categorical_inputs.size(0) >= 10:
                        try:
                            toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                            applied_ranking_loss = ranking_criterion(applied_pred.squeeze(), applied_labels.squeeze())
                            rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                            # Clamp ranking losses
                            toured_ranking_loss = torch.clamp(toured_ranking_loss, -100, 100)
                            applied_ranking_loss = torch.clamp(applied_ranking_loss, -100, 100)
                            rented_ranking_loss = torch.clamp(rented_ranking_loss, -100, 100)

                            ranking_loss = toured_weight * toured_ranking_loss + \
                                           applied_weight * applied_ranking_loss + \
                                           rented_weight * rented_ranking_loss

                            # Combine BCE and ranking loss
                            loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                        except Exception as e:
                            print(f"Ranking loss failed: {str(e)}, using BCE only")
                            loss = bce_loss
                    else:
                        # Use only BCE if batch is too small
                        loss = bce_loss

                    # Final sanity check on loss value
                    if not torch.isfinite(loss):
                        print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                        loss = torch.tensor(10.0, device=device)  # Safe fallback

                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item() * gradient_accumulation_steps
                if verbose and hasattr(train_iter, 'set_postfix'):
                    train_iter.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                print(f"Batch shapes: cat={categorical_inputs.shape}, num={numerical_inputs.shape}")
                # Skip this batch and continue
                continue

        train_loss /= max(1, len(train_loader))
        history['train_loss'].append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        valid_batches = 0

        # Use tqdm only if verbose mode is on
        if verbose:
            valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")
        else:
            valid_iter = valid_loader

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
                    # Use evaluation mode (actual filtering between stages)
                    toured_pred, applied_pred, rented_pred, toured_idx, applied_idx, rented_idx = model(
                        categorical_inputs, numerical_inputs, lead_ids, is_training=False
                    )

                    # Simple validation loss without weighting for simplicity
                    toured_loss = toured_criterion(toured_pred, toured_labels).mean()
                    applied_loss = applied_criterion(applied_pred, applied_labels).mean()
                    rented_loss = rented_criterion(rented_pred, rented_labels).mean()

                    batch_loss = (toured_weight * toured_loss +
                                  applied_weight * applied_loss +
                                  rented_weight * rented_loss)

                    # Clamp validation loss too
                    batch_loss = torch.clamp(batch_loss, 0, 100)

                    val_loss += batch_loss.item()
                    valid_batches += 1
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                # Skip this batch and continue
                continue

        val_loss /= max(1, valid_batches)
        history['val_loss'].append(val_loss)

        # Print progress
        if verbose or epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

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
                print(f"Early stopping after {epoch + 1} epochs")
                break

    # Load best state
    try:
        model.load_state_dict(torch.load(model_save_path))
    except Exception as e:
        print(f"Error loading best model state: {str(e)}")
        print("Continuing with current model state...")

    return model, history


# Function for cascade ranking
def cascade_rank_leads(model, dataloader, device='cuda', silent=True):
    """
    Uses the cascaded model to rank leads through the funnel
    Returns dict with selected IDs and scores for each stage
    """
    model.eval()

    all_lead_ids = []
    all_toured_preds = []
    all_applied_preds = []
    all_rented_preds = []
    all_toured_indices = []
    all_applied_indices = []
    all_rented_indices = []

    with torch.no_grad():
        iterator = dataloader if silent else tqdm(dataloader, desc="Ranking leads")
        for batch in iterator:
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
            lead_ids = batch[-1].cpu().numpy()

            # Skip empty batches
            if cat_in.size(0) == 0:
                continue

            try:
                # Forward pass with filtering between stages
                toured_pred, applied_pred, rented_pred, toured_idx, applied_idx, rented_idx = model(
                    cat_in, num_in, is_training=False
                )

                # Store predictions and indices
                all_lead_ids.extend(lead_ids)
                all_toured_preds.extend(toured_pred.cpu().numpy())
                all_applied_preds.extend(applied_pred.cpu().numpy())
                all_rented_preds.extend(rented_pred.cpu().numpy())

                # Convert indices to CPU for collection
                toured_idx_cpu = toured_idx.cpu().numpy()
                if len(toured_idx_cpu) > 0:  # Only collect if indices exist
                    all_toured_indices.extend(lead_ids[toured_idx_cpu])

                applied_idx_cpu = applied_idx.cpu().numpy()
                if len(applied_idx_cpu) > 0:  # Only collect if indices exist
                    all_applied_indices.extend(lead_ids[applied_idx_cpu])

                rented_idx_cpu = rented_idx.cpu().numpy()
                if len(rented_idx_cpu) > 0:  # Only collect if indices exist
                    all_rented_indices.extend(lead_ids[rented_idx_cpu])
            except Exception as e:
                print(f"Error in ranking batch: {str(e)}")
                # Skip this batch and continue
                continue

    # Convert to arrays
    all_lead_ids = np.array(all_lead_ids)
    all_toured_preds = np.array(all_toured_preds).flatten()
    all_applied_preds = np.array(all_applied_preds).flatten()
    all_rented_preds = np.array(all_rented_preds).flatten()

    # Handle empty results
    if len(all_lead_ids) == 0:
        return {
            'toured': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])},
            'applied': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])},
            'rented': {'selected': np.array([]), 'scores': np.array([]), 'excluded': np.array([])}
        }

    # Get unique indices (in case of overlap between batches)
    toured_selected_ids = np.unique(all_toured_indices) if len(all_toured_indices) > 0 else np.array([])
    applied_selected_ids = np.unique(all_applied_indices) if len(all_applied_indices) > 0 else np.array([])
    rented_selected_ids = np.unique(all_rented_indices) if len(all_rented_indices) > 0 else np.array([])

    # Print summary
    print(f"\nRanking Summary:")
    print(f"Selected {len(toured_selected_ids)} leads for touring out of {len(all_lead_ids)}")
    print(
        f"Selected {len(applied_selected_ids)} leads for applications out of {len(toured_selected_ids) if len(toured_selected_ids) > 0 else 0}")
    print(
        f"Selected {len(rented_selected_ids)} leads for rental out of {len(applied_selected_ids) if len(applied_selected_ids) > 0 else 0}")

    # Get scores for selected IDs - handle empty selections
    toured_scores = all_toured_preds[np.isin(all_lead_ids, toured_selected_ids)] if len(
        toured_selected_ids) > 0 else np.array([])
    applied_scores = all_applied_preds[np.isin(all_lead_ids, applied_selected_ids)] if len(
        applied_selected_ids) > 0 else np.array([])
    rented_scores = all_rented_preds[np.isin(all_lead_ids, rented_selected_ids)] if len(
        rented_selected_ids) > 0 else np.array([])

    # Excluded IDs - handle empty selections
    toured_excluded_ids = np.setdiff1d(all_lead_ids, toured_selected_ids) if len(
        toured_selected_ids) > 0 else all_lead_ids
    applied_excluded_ids = np.setdiff1d(toured_selected_ids, applied_selected_ids) if len(
        toured_selected_ids) > 0 and len(applied_selected_ids) > 0 else toured_selected_ids
    rented_excluded_ids = np.setdiff1d(applied_selected_ids, rented_selected_ids) if len(
        applied_selected_ids) > 0 and len(rented_selected_ids) > 0 else applied_selected_ids

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