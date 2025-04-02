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
        # Replace linear layers with proper embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(min(cat_dim, 100000), emb_dim, sparse=False) 
            for cat_dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        self.numerical_bn = nn.BatchNorm1d(numerical_dim) if numerical_dim > 0 else None
        
        # Store dimensions for validation during forward pass
        self.categorical_dims = categorical_dims
        
        # Print embedding info
        print(f"Created {len(self.categorical_embeddings)} embedding layers")
        total_params = sum(dim * emb_dim for dim, emb_dim in zip(
            [min(d, 100000) for d in categorical_dims], 
            embedding_dims
        ))
        print(f"Total embedding parameters: {total_params:,}")

    def forward(self, categorical_inputs, numerical_inputs):
        embedded_features = []
        
        if len(self.categorical_embeddings) > 0:
            for i, embedding_layer in enumerate(self.categorical_embeddings):
                try:
                    # Convert to long tensor for embedding lookup
                    cat_feature = categorical_inputs[:, i].long()
                    
                    # Safety check for out-of-bounds indices
                    vocab_size = self.categorical_dims[i]
                    max_idx = min(vocab_size - 1, 99999)  # Cap at 100K - 1
                    
                    # Clamp indices to valid range
                    cat_feature = torch.clamp(cat_feature, 0, max_idx)
                    
                    # Get embeddings
                    embedded = embedding_layer(cat_feature)
                    embedded_features.append(embedded)
                    
                except Exception as e:
                    # If embedding fails, create a zero tensor of appropriate shape
                    emb_dim = embedding_layer.embedding_dim
                    zero_emb = torch.zeros((categorical_inputs.size(0), emb_dim), 
                                          device=categorical_inputs.device)
                    embedded_features.append(zero_emb)
                    print(f"Warning: Embedding error for feature {i}: {str(e)}")

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
        # Handle empty inputs
        if len(scores) == 0 or len(targets) == 0:
            # Return a valid loss value of 0 with gradients
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
            
        # Reshape if needed
        if scores.dim() > 1:
            scores = scores.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Safety check for infinite values
        if torch.any(torch.isinf(scores)) or torch.any(torch.isnan(scores)):
            # Replace inf/nan values with safe values
            scores = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
            
        # Add a small epsilon to prevent numerical issues
        epsilon = 1e-7
        
        # Sort targets in descending order with stability
        try:
            # Safer sorting with a small random noise to break ties consistently
            random_noise = torch.randn_like(targets) * 1e-8
            sorted_targets, indices = torch.sort(targets + random_noise, descending=True, dim=0)
            
            # Reorder scores according to target sorting
            ordered_scores = torch.gather(scores, 0, indices)
            
            # Add a small constant to stabilize scores
            ordered_scores = ordered_scores + epsilon
            
            # Apply log_softmax with temperature scaling and clamp extreme values
            temperature = 0.5  # Increased from 0.3 for more stability
            scores_softmax = F.log_softmax(ordered_scores / temperature, dim=0)
            
            # Stricter clamping to prevent extreme values
            scores_softmax = torch.clamp(scores_softmax, min=-3, max=3)
            
            # Compute loss
            loss = -torch.mean(scores_softmax)
            
            # Safety check on the final loss
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.3, device=scores.device, requires_grad=True)
                
            return torch.clamp(loss, min=-3, max=3)
            
        except Exception as e:
            # If anything fails, return a safe fallback value
            print(f"ListMLELoss error: {str(e)}, returning fallback value")
            return torch.tensor(0.3, device=scores.device, requires_grad=True)


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

        # 1. Check if embedding dimensions need adjustment
        # Cap embedding dimensions to prevent instability with large vocabularies
        modified_embedding_dims = []
        for dim in embedding_dims:
            # Cap embeddings at 25 dimensions for stability
            modified_embedding_dims.append(min(25, dim))
        
        # 1. Embedding layer
        self.embedding_layer = EmbeddingLayer(categorical_dims, modified_embedding_dims, numerical_dim)

        # 2. Projection to transformer dimension
        total_emb_dim = sum(modified_embedding_dims) + numerical_dim
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
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout),  # Standard size
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
        """More conservative weight initialization to prevent learning issues"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use more conservative initialization with smaller weights
                nn.init.xavier_uniform_(module.weight.data, gain=0.2)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Embedding):
                # Use more conservative embedding initialization
                nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

    def forward(self, categorical_inputs, numerical_inputs, lead_ids=None, is_training=True):
        """
        Forward pass with cascading stages and filtering between them.
        During training: process all leads through all stages but with gradient tracking
        During inference: filter at each stage before proceeding
        """
        batch_size = categorical_inputs.size(0)
        device = categorical_inputs.device

        # Ensure inputs have the correct dtype
        categorical_inputs = categorical_inputs.long()
        numerical_inputs = numerical_inputs.float()

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
            # Make sure we're sorting a leaf tensor that requires_grad
            sortable_scores = toured_pred.squeeze() + noise
            _, toured_indices = torch.topk(sortable_scores, k_toured)

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
                        
                    # Add noise to break ties
                    noise = torch.randn_like(applied_pred.squeeze()[toured_indices]) * 1e-6
                    # Make sure we're sorting a leaf tensor that requires_grad
                    sortable_scores = applied_pred.squeeze()[toured_indices] + noise
                    _, applied_indices_local = torch.topk(sortable_scores, k_applied)
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
                    # Make sure we're sorting a leaf tensor that requires_grad
                    sortable_scores = rented_pred.squeeze()[applied_indices] + noise
                    _, rented_indices_local = torch.topk(sortable_scores, k_rented)
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
                    # Make sure we're sorting a leaf tensor
                    sortable_scores = applied_pred_subset.squeeze() + noise
                    _, applied_indices_local = torch.topk(sortable_scores, k_applied)
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
                    # Make sure we're sorting a leaf tensor
                    sortable_scores = rented_pred_subset.squeeze() + noise
                    _, rented_indices_local = torch.topk(sortable_scores, k_rented)
                rented_indices = applied_indices[rented_indices_local]

        # Add safeguards to ensure valid predictions
        # Add small epsilon to prevent exact 0s and 1s
        epsilon = 1e-6
        toured_pred = torch.clamp(toured_pred, epsilon, 1.0 - epsilon)
        applied_pred = torch.clamp(applied_pred, epsilon, 1.0 - epsilon)
        rented_pred = torch.clamp(rented_pred, epsilon, 1.0 - epsilon)

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
    # Standard classification losses with reduction='none' for more control
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
    
    # WARM-UP PHASE: Use BCE only for first 2 epochs with lower learning rate
    original_lr = optimizer.param_groups[0]['lr']
    warm_up_epochs = 2
    
    for epoch in range(num_epochs):
        # During warm-up, use lower learning rate and simpler loss
        is_warmup = epoch < warm_up_epochs
        if is_warmup:
            # Use 10% of the learning rate during warm-up
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr * 0.1
            
            if verbose and epoch == 0:
                print("\nWARM-UP PHASE: Using reduced learning rate and BCE loss only")
        elif epoch == warm_up_epochs:
            # Gradual learning rate increase instead of immediate jump
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr * 0.2  # Start at 20% instead of full
            if verbose:
                print("\nWARM-UP COMPLETE: Using reduced learning rate (20%) with full loss function")
        elif epoch == warm_up_epochs + 1:
            # Further increase on the next epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr * 0.5
            if verbose:
                print("Increasing learning rate to 50%")
        elif epoch == warm_up_epochs + 2:
            # Full learning rate only after 2 more epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr
            if verbose:
                print("Restored original learning rate")
        
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
                        toured_losses = torch.clamp(toured_losses, 0, 5)
                        toured_loss = toured_losses.mean()

                        # For applied, weight higher for toured selected
                        applied_losses = applied_criterion(applied_pred, applied_labels)
                        applied_losses = torch.clamp(applied_losses, 0, 5)

                        toured_mask = torch.ones_like(applied_losses, device=device)  # Default weight 1.0
                        if len(toured_idx) > 0 and not is_warmup:  # Only use masks after warm-up
                            toured_mask[toured_idx] = 2.0  # Higher weight for selected leads
                            # Set mask for non-selected items
                            non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                            toured_mask[non_selected_mask] = 0.5
                        applied_loss = (applied_losses * toured_mask).mean()

                        # For rented, weight higher for applied selected
                        rented_losses = rented_criterion(rented_pred, rented_labels)
                        rented_losses = torch.clamp(rented_losses, 0, 5)

                        applied_mask = torch.ones_like(rented_losses, device=device)  # Default weight 1.0
                        if len(applied_idx) > 0 and not is_warmup:  # Only use masks after warm-up
                            applied_mask[applied_idx] = 2.0
                            # Set mask for non-selected items
                            non_selected_mask = ~torch.isin(torch.arange(len(applied_mask), device=device), applied_idx)
                            applied_mask[non_selected_mask] = 0.5
                        rented_loss = (rented_losses * applied_mask).mean()

                        # Standard BCE loss
                        bce_loss = toured_weight * toured_loss + \
                                   applied_weight * applied_loss + \
                                   rented_weight * rented_loss

                        # Add Ranking Loss - only after warm-up and with enough batch size
                        if not is_warmup and categorical_inputs.size(0) >= 10:
                            try:
                                toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                                applied_ranking_loss = ranking_criterion(applied_pred.squeeze(),
                                                                         applied_labels.squeeze())
                                rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                                # Clamp ranking losses to safer values
                                toured_ranking_loss = torch.clamp(toured_ranking_loss, -3, 3)
                                applied_ranking_loss = torch.clamp(applied_ranking_loss, -3, 3)
                                rented_ranking_loss = torch.clamp(rented_ranking_loss, -3, 3)

                                ranking_loss = toured_weight * toured_ranking_loss + \
                                               applied_weight * applied_ranking_loss + \
                                               rented_weight * rented_ranking_loss

                                # Combine BCE and ranking loss
                                # loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                                loss = bce_loss # Temporarily use only BCE
                            except Exception as e:
                                # Fallback if ranking loss fails
                                if verbose:
                                    print(f"Ranking loss failed: {str(e)}, using BCE only")
                                loss = bce_loss
                        else:
                            # Use only BCE during warm-up or if batch is too small
                            loss = bce_loss

                        # Final safety check on loss value - more conservative clamping
                        if not torch.isfinite(loss) or loss > 5:
                            if verbose:
                                print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                            loss = torch.tensor(0.5, device=device, requires_grad=True)  # Safer fallback value

                        # gradient accumulation
                        loss = loss / gradient_accumulation_steps

                    # Skip backward pass if loss is invalid
                    if not torch.isfinite(loss):
                        if verbose:
                            print(f"WARNING: Skipping batch {batch_idx} due to non-finite loss: {loss.item()}")
                        # Need to potentially zero_grad if accumulation happened before invalid loss
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                             optimizer.zero_grad()
                        continue # Skip the rest of the loop for this batch

                    scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Much tighter clipping from 0.5
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
                    toured_losses = torch.clamp(toured_losses, 0, 5)  # More conservative clamping
                    toured_loss = toured_losses.mean()

                    applied_losses = applied_criterion(applied_pred, applied_labels)
                    applied_losses = torch.clamp(applied_losses, 0, 5)  # More conservative clamping

                    toured_mask = torch.ones_like(applied_losses, device=device)  # Default weight 1.0
                    if len(toured_idx) > 0 and not is_warmup:  # Only use masks after warm-up
                        toured_mask[toured_idx] = 2.0
                        non_selected_mask = ~torch.isin(torch.arange(len(toured_mask), device=device), toured_idx)
                        toured_mask[non_selected_mask] = 0.5
                    applied_loss = (applied_losses * toured_mask).mean()

                    rented_losses = rented_criterion(rented_pred, rented_labels)
                    rented_losses = torch.clamp(rented_losses, 0, 5)  # More conservative clamping

                    applied_mask = torch.ones_like(rented_losses, device=device)  # Default weight 1.0
                    if len(applied_idx) > 0 and not is_warmup:  # Only use masks after warm-up
                        applied_mask[applied_idx] = 2.0
                        non_selected_mask = ~torch.isin(torch.arange(len(applied_mask), device=device), applied_idx)
                        applied_mask[non_selected_mask] = 0.5
                    rented_loss = (rented_losses * applied_mask).mean()

                    # Standard BCE loss
                    bce_loss = toured_weight * toured_loss + \
                               applied_weight * applied_loss + \
                               rented_weight * rented_loss

                    # Add Ranking Loss - only after warm-up period
                    if not is_warmup and categorical_inputs.size(0) >= 10:
                        try:
                            toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                            applied_ranking_loss = ranking_criterion(applied_pred.squeeze(), applied_labels.squeeze())
                            rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                            # Clamp ranking losses to safer values
                            toured_ranking_loss = torch.clamp(toured_ranking_loss, -3, 3)
                            applied_ranking_loss = torch.clamp(applied_ranking_loss, -3, 3)
                            rented_ranking_loss = torch.clamp(rented_ranking_loss, -3, 3)

                            ranking_loss = toured_weight * toured_ranking_loss + \
                                           applied_weight * applied_ranking_loss + \
                                           rented_weight * rented_ranking_loss

                            # Combine BCE and ranking loss
                            # loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss
                            loss = bce_loss # Temporarily use only BCE
                        except Exception as e:
                            if verbose:
                                print(f"Ranking loss failed: {str(e)}, using BCE only")
                            loss = bce_loss
                    else:
                        # Use only BCE during warm-up or if batch is too small
                        loss = bce_loss

                    # Final safety check
                    if not torch.isfinite(loss) or loss > 5:  # More conservative threshold
                        if verbose:
                            print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                        loss = torch.tensor(0.5, device=device, requires_grad=True)  # Safer fallback

                    loss = loss / gradient_accumulation_steps

                    # Skip backward pass if loss is invalid
                    if not torch.isfinite(loss):
                        if verbose:
                            print(f"WARNING: Skipping batch {batch_idx} due to non-finite loss: {loss.item()}")
                        # Need to potentially zero_grad if accumulation happened before invalid loss
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.zero_grad()
                        continue # Skip the rest of the loop for this batch

                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Much tighter clipping
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
                        batch_loss = torch.tensor(3.0, device=device)

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