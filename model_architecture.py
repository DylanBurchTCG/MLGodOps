import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv
import os


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

        # Add a small epsilon to prevent numerical issues
        epsilon = 1e-10
        
        # Sort targets in descending order
        sorted_targets, indices = torch.sort(targets, descending=True, dim=0)

        # Reorder scores according to target sorting
        ordered_scores = torch.gather(scores, 0, indices)

        # Numerically stable implementation
        # Apply log_softmax with temperature scaling to avoid extreme values
        temperature = 0.1
        scores_softmax = F.log_softmax(ordered_scores / temperature, dim=0)
        
        # Compute loss with clipping to prevent extreme values
        loss = -torch.mean(torch.clamp(scores_softmax, min=-10, max=10))

        # Return a valid scalar loss value
        return torch.clamp(loss, min=-5, max=5)


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
                 rented_k=250,
                 use_percentages=False,  # NEW: Flag to use percentages instead of fixed counts
                 toured_pct=None,       # NEW: Percentage of leads to select for touring stage
                 applied_pct=None,      # NEW: Percentage of toured leads to select for applying stage
                 rented_pct=None):      # NEW: Percentage of applied leads to select for renting stage
        super().__init__()

        # Stage filtering parameters
        self.toured_k = toured_k
        self.applied_k = applied_k
        self.rented_k = rented_k
        
        # NEW: Percentage-based selection parameters
        self.use_percentages = use_percentages
        self.toured_pct = toured_pct if toured_pct is not None else 0.5  # Default 50%
        self.applied_pct = applied_pct if applied_pct is not None else 0.5  # Default 50%
        self.rented_pct = rented_pct if rented_pct is not None else 0.5  # Default 50%
        
        print(f"Initializing MultiTaskCascadedLeadFunnelModel with: ")
        print(f"  - {len(categorical_dims)} categorical features")
        print(f"  - {numerical_dim} numerical features")
        print(f"  - Transformer dim: {transformer_dim}, heads: {num_heads}")
        if use_percentages:
            print(f"  - Selection by percentage: {self.toured_pct*100:.1f}% -> {self.applied_pct*100:.1f}% -> {self.rented_pct*100:.1f}%")
        else:
            print(f"  - Selection flow: {toured_k} -> {applied_k} -> {rented_k}")

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
        # ENHANCED: Deeper transformer path for toured stage (3 blocks instead of 1)
        self.transformer_toured = nn.Sequential(
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout),
            TabularTransformerBlock(transformer_dim, num_heads*2, ff_dim*2, dropout=dropout),  # Wider block
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout)
        )
        
        # Add a special feature enhancement layer for toured prediction
        self.toured_feature_enhancement = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim*2),
            nn.LayerNorm(transformer_dim*2),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),  # Higher dropout for this path
            nn.Linear(transformer_dim*2, transformer_dim)
        )
        
        # Regular transformers for other stages
        self.transformer_applied = TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)
        self.transformer_rented = TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)

        # 5. Prediction heads for each stage
        # ENHANCED: Deeper prediction head for toured stage
        toured_hidden_dims = [dim * 2 for dim in head_hidden_dims]  # Double the neurons
        self.toured_head = nn.Sequential(
            nn.Linear(transformer_dim, toured_hidden_dims[0]),
            nn.BatchNorm1d(toured_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),  # Higher dropout
            nn.Linear(toured_hidden_dims[0], toured_hidden_dims[0] // 2),
            nn.BatchNorm1d(toured_hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.05),
            nn.Linear(toured_hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )
        
        # Regular prediction heads for other stages
        self.applied_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.rented_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Careful weight initialization to prevent learning issues"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight.data, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for LayerNorm
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)

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

        # Stage 1: Toured prediction with enhanced path
        if isinstance(self.transformer_toured, nn.Sequential):
            toured_features = self.transformer_toured(shared_features)
        else:
            toured_features = self.transformer_toured(shared_features)
            
        # Apply special feature enhancement for toured stage
        toured_features = self.toured_feature_enhancement(toured_features)
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

        # NEW: Determine k for toured stage - either fixed count or percentage
        if self.use_percentages:
            k_toured = max(1, int(batch_size * self.toured_pct))
        else:
            k_toured = min(self.toured_k, batch_size)
            
        if toured_pred.squeeze().dim() == 0:  # Handle scalar case
            toured_indices = torch.tensor([0], device=device)
        else:
            # Apply slight noise to break ties randomly
            noise = torch.randn_like(toured_pred.squeeze()) * 1e-6
            _, toured_indices = torch.topk(toured_pred.squeeze() + noise, k_toured)

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
                    # NEW: Determine k for applied stage - either fixed count or percentage
                    if self.use_percentages:
                        k_applied = max(1, int(len(toured_indices) * self.applied_pct))
                    else:
                        k_applied = min(self.applied_k, len(toured_indices))
                        
                    # Add noise to break ties randomly
                    noise = torch.randn_like(applied_pred.squeeze()[toured_indices]) * 1e-6
                    _, applied_indices_local = torch.topk(
                        applied_pred.squeeze()[toured_indices] + noise, 
                        k_applied
                    )
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
                    # NEW: Determine k for rented stage - either fixed count or percentage
                    if self.use_percentages:
                        k_rented = max(1, int(len(applied_indices) * self.rented_pct))
                    else:
                        k_rented = min(self.rented_k, len(applied_indices))
                        
                    # Add noise to break ties
                    noise = torch.randn_like(rented_pred.squeeze()[applied_indices]) * 1e-6
                    _, rented_indices_local = torch.topk(
                        rented_pred.squeeze()[applied_indices] + noise, 
                        k_rented
                    )
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
                    # NEW: Determine k for applied stage - either fixed count or percentage
                    if self.use_percentages:
                        k_applied = max(1, int(len(toured_indices) * self.applied_pct))
                    else:
                        k_applied = min(self.applied_k, len(toured_indices))
                        
                    # Add noise to break ties
                    noise = torch.randn_like(applied_pred_subset.squeeze()) * 1e-6
                    _, applied_indices_local = torch.topk(
                        applied_pred_subset.squeeze() + noise, 
                        k_applied
                    )
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
                    # NEW: Determine k for rented stage - either fixed count or percentage
                    if self.use_percentages:
                        k_rented = max(1, int(len(applied_indices) * self.rented_pct))
                    else:
                        k_rented = min(self.rented_k, len(applied_indices))
                        
                    # Add noise to break ties
                    noise = torch.randn_like(rented_pred_subset.squeeze()) * 1e-6
                    _, rented_indices_local = torch.topk(
                        rented_pred_subset.squeeze() + noise, 
                        k_rented
                    )
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
                         verbose=True,
                         epoch_csv_path=None):  # New parameter to save per-epoch metrics to CSV
    """
    Training loop for cascaded model with ranking loss
    """
    # Create CSV file for per-epoch metrics if requested
    csv_file = None
    csv_writer = None
    if epoch_csv_path:
        # Create directories if needed
        os.makedirs(os.path.dirname(epoch_csv_path), exist_ok=True)
        
        # Open file and create writer
        csv_file = open(epoch_csv_path, 'w', newline='')
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'learning_rate',
            'toured_auc', 'applied_auc', 'rented_auc',
            'toured_apr', 'applied_apr', 'rented_apr',
            'toured_p50', 'applied_p50', 'rented_p50'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
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
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Print training configuration
    if verbose:
        print(f"\nTraining cascade model for {num_epochs} epochs")
        print(f"Device: {device}, Mixed precision: {mixed_precision}")
        print(f"Batch size: {train_loader.batch_size}, Grad accumulation: {gradient_accumulation_steps}")
        print(f"Loss weights: toured={toured_weight}, applied={applied_weight}, rented={rented_weight}")
        print(f"Ranking loss weight: {ranking_weight}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("-" * 60)

    for epoch in range(num_epochs):
        # -------- TRAINING --------
        model.train()
        train_loss = 0.0
        num_batches = 0
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

                        # Add epsilon to prevent exact 0s and 1s
                        epsilon = 1e-7
                        toured_pred = torch.clamp(toured_pred, epsilon, 1 - epsilon)
                        applied_pred = torch.clamp(applied_pred, epsilon, 1 - epsilon)
                        rented_pred = torch.clamp(rented_pred, epsilon, 1 - epsilon)

                        # Calculate BCE losses with focus on selected indices
                        toured_losses = toured_criterion(toured_pred, toured_labels)

                        # Add loss clipping to prevent extremely large loss values
                        toured_losses = torch.clamp(toured_losses, 0, 10)
                        toured_loss = toured_losses.mean()

                        # For applied, weight higher for toured selected
                        applied_losses = applied_criterion(applied_pred, applied_labels)
                        applied_losses = torch.clamp(applied_losses, 0, 10)

                        toured_mask = torch.zeros_like(applied_losses, device=device)
                        if len(toured_idx) > 0:  # Only set weights if we have selected indices
                            toured_mask[toured_idx] = 2.0  # Higher weight for selected leads
                            # Set mask for non-selected items
                            non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                            toured_mask[non_selected_mask] = 0.5
                        applied_loss = (applied_losses * (toured_mask + 0.1)).mean()

                        # For rented, weight higher for applied selected
                        rented_losses = rented_criterion(rented_pred, rented_labels)
                        rented_losses = torch.clamp(rented_losses, 0, 10)

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
                                toured_ranking_loss = torch.clamp(toured_ranking_loss, -5, 5)
                                applied_ranking_loss = torch.clamp(applied_ranking_loss, -5, 5)
                                rented_ranking_loss = torch.clamp(rented_ranking_loss, -5, 5)

                                ranking_loss = toured_weight * toured_ranking_loss + \
                                               applied_weight * applied_ranking_loss + \
                                               rented_weight * rented_ranking_loss

                                # Combine BCE and ranking loss
                                loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                            except Exception as e:
                                # Fallback if ranking loss fails
                                if verbose:
                                    print(f"Ranking loss failed: {str(e)}, using BCE only")
                                loss = bce_loss
                        else:
                            # Use only BCE if batch is too small
                            loss = bce_loss

                        # Final sanity check on loss value
                        if not torch.isfinite(loss) or loss > 10:
                            if verbose:
                                print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                            loss = torch.tensor(1.0, device=device)  # Safe fallback

                        # gradient accumulation
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        # Reduced gradient clipping threshold for more stability
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # No mixed precision - implement similar logic
                    toured_pred, applied_pred, rented_pred, toured_idx, applied_idx, rented_idx = model(
                        categorical_inputs, numerical_inputs, lead_ids, is_training=True
                    )

                    # Add epsilon to prevent exact 0s and 1s
                    epsilon = 1e-7
                    toured_pred = torch.clamp(toured_pred, epsilon, 1 - epsilon)
                    applied_pred = torch.clamp(applied_pred, epsilon, 1 - epsilon)
                    rented_pred = torch.clamp(rented_pred, epsilon, 1 - epsilon)

                    # Calculate BCE losses
                    toured_losses = toured_criterion(toured_pred, toured_labels)
                    toured_losses = torch.clamp(toured_losses, 0, 10)
                    toured_loss = toured_losses.mean()

                    applied_losses = applied_criterion(applied_pred, applied_labels)
                    applied_losses = torch.clamp(applied_losses, 0, 10)

                    toured_mask = torch.zeros_like(applied_losses, device=device)
                    if len(toured_idx) > 0:
                        toured_mask[toured_idx] = 2.0
                        non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                        toured_mask[non_selected_mask] = 0.5
                    applied_loss = (applied_losses * (toured_mask + 0.1)).mean()

                    rented_losses = rented_criterion(rented_pred, rented_labels)
                    rented_losses = torch.clamp(rented_losses, 0, 10)

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
                            toured_ranking_loss = torch.clamp(toured_ranking_loss, -5, 5)
                            applied_ranking_loss = torch.clamp(applied_ranking_loss, -5, 5)
                            rented_ranking_loss = torch.clamp(rented_ranking_loss, -5, 5)

                            ranking_loss = toured_weight * toured_ranking_loss + \
                                           applied_weight * applied_ranking_loss + \
                                           rented_weight * rented_ranking_loss

                            # Combine BCE and ranking loss
                            loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                        except Exception as e:
                            if verbose:
                                print(f"Ranking loss failed: {str(e)}, using BCE only")
                            loss = bce_loss
                    else:
                        # Use only BCE if batch is too small
                        loss = bce_loss

                    # Final sanity check on loss value
                    if not torch.isfinite(loss) or loss > 10:
                        if verbose:
                            print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                        loss = torch.tensor(1.0, device=device)  # Safe fallback

                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar with actual loss values
                if verbose and hasattr(train_iter, 'set_postfix'):
                    train_iter.set_postfix({
                        'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                        'toured': f"{toured_loss.item():.4f}",
                        'applied': f"{applied_loss.item():.4f}",
                        'rented': f"{rented_loss.item():.4f}"
                    })
                
                num_batches += 1

            except Exception as e:
                if verbose:
                    print(f"Error in training batch {batch_idx}: {str(e)}")
                    print(f"Batch shapes: cat={categorical_inputs.shape}, num={numerical_inputs.shape}")
                # Skip this batch and continue
                continue

        train_loss /= max(1, num_batches)
        history['train_loss'].append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        valid_batches = 0
        all_toured_preds = []
        all_toured_true = []
        all_applied_preds = []
        all_applied_true = []
        all_rented_preds = []
        all_rented_true = []

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

                    # Add epsilon to predictions for numerical stability
                    epsilon = 1e-7
                    toured_pred = torch.clamp(toured_pred, epsilon, 1 - epsilon)
                    applied_pred = torch.clamp(applied_pred, epsilon, 1 - epsilon)
                    rented_pred = torch.clamp(rented_pred, epsilon, 1 - epsilon)

                    # Simple validation loss without weighting for simplicity
                    toured_loss = toured_criterion(toured_pred, toured_labels).mean()
                    applied_loss = applied_criterion(applied_pred, applied_labels).mean()
                    rented_loss = rented_criterion(rented_pred, rented_labels).mean()

                    batch_loss = (toured_weight * toured_loss +
                                  applied_weight * applied_loss +
                                  rented_weight * rented_loss)

                    # Clamp validation loss
                    batch_loss = torch.clamp(batch_loss, 0, 10)

                    val_loss += batch_loss.item()
                    valid_batches += 1
                    
                    # Store predictions for metrics calculation
                    all_toured_preds.append(toured_pred.cpu().numpy())
                    all_toured_true.append(toured_labels.cpu().numpy())
                    all_applied_preds.append(applied_pred.cpu().numpy())
                    all_applied_true.append(applied_labels.cpu().numpy())
                    all_rented_preds.append(rented_pred.cpu().numpy())
                    all_rented_true.append(rented_labels.cpu().numpy())
                    
            except Exception as e:
                if verbose:
                    print(f"Error in validation batch: {str(e)}")
                # Skip this batch and continue
                continue

        val_loss /= max(1, valid_batches)
        history['val_loss'].append(val_loss)
        
        # Calculate metrics
        try:
            # Combine predictions
            all_toured_preds = np.vstack(all_toured_preds)
            all_toured_true = np.vstack(all_toured_true)
            all_applied_preds = np.vstack(all_applied_preds)
            all_applied_true = np.vstack(all_applied_true)
            all_rented_preds = np.vstack(all_rented_preds)
            all_rented_true = np.vstack(all_rented_true)
            
            # AUC scores
            from sklearn.metrics import roc_auc_score, average_precision_score
            toured_auc = roc_auc_score(all_toured_true, all_toured_preds)
            applied_auc = roc_auc_score(all_applied_true, all_applied_preds)
            rented_auc = roc_auc_score(all_rented_true, all_rented_preds)
            
            # APR scores
            toured_apr = average_precision_score(all_toured_true, all_toured_preds)
            applied_apr = average_precision_score(all_applied_true, all_applied_preds)
            rented_apr = average_precision_score(all_rented_true, all_rented_preds)
            
            # Precision@k
            k = 50
            toured_p50 = precision_at_k(all_toured_true.flatten(), all_toured_preds.flatten(), k)
            applied_p50 = precision_at_k(all_applied_true.flatten(), all_applied_preds.flatten(), k)
            rented_p50 = precision_at_k(all_rented_true.flatten(), all_rented_preds.flatten(), k)
            
            # Add metrics to history
            if 'toured_auc' not in history:
                history['toured_auc'] = []
                history['applied_auc'] = []
                history['rented_auc'] = []
                history['toured_apr'] = []
                history['applied_apr'] = []
                history['rented_apr'] = []
                history['toured_p50'] = []
                history['applied_p50'] = []
                history['rented_p50'] = []
                
            history['toured_auc'].append(toured_auc)
            history['applied_auc'].append(applied_auc)
            history['rented_auc'].append(rented_auc)
            history['toured_apr'].append(toured_apr)
            history['applied_apr'].append(applied_apr)
            history['rented_apr'].append(rented_apr)
            history['toured_p50'].append(toured_p50)
            history['applied_p50'].append(applied_p50)
            history['rented_p50'].append(rented_p50)
            
            has_metrics = True
            
            # Write to CSV if requested
            if csv_writer:
                csv_writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    'toured_auc': toured_auc,
                    'applied_auc': applied_auc,
                    'rented_auc': rented_auc,
                    'toured_apr': toured_apr,
                    'applied_apr': applied_apr,
                    'rented_apr': rented_apr,
                    'toured_p50': toured_p50,
                    'applied_p50': applied_p50,
                    'rented_p50': rented_p50
                })
                # Flush to ensure data is written immediately
                csv_file.flush()
                
        except Exception as e:
            if verbose:
                print(f"Error calculating metrics: {str(e)}")
            has_metrics = False

        # Print progress
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            if has_metrics:
                print(f"\nMetrics:")
                print(f"  Toured:  AUC={toured_auc:.4f}  APR={toured_apr:.4f}  P@50={toured_p50:.4f}")
                print(f"  Applied: AUC={applied_auc:.4f}  APR={applied_apr:.4f}  P@50={applied_p50:.4f}")
                print(f"  Rented:  AUC={rented_auc:.4f}  APR={rented_apr:.4f}  P@50={rented_p50:.4f}")
            print("-" * 60)

        # Step LR scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stopping_counter = 0
            if verbose:
                print(f"Model improved! Saved checkpoint at epoch {epoch + 1}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {epoch + 1} epochs")
                break

    # Close CSV file if it's open
    if csv_file:
        csv_file.close()
        if verbose:
            print(f"Per-epoch metrics saved to {epoch_csv_path}")

    # Load best state
    try:
        model.load_state_dict(torch.load(model_save_path))
        if verbose:
            print(f"Loaded best model state from {model_save_path}")
    except Exception as e:
        if verbose:
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