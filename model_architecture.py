import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv
import os


def safe_loss(loss, max_value=2.0, fallback=1.0, name=""):
    """
    Helper function to handle NaN, Inf, or extremely large loss values.
    Returns a clamped value or fallback if the loss is problematic.
    
    Args:
        loss: The loss tensor
        max_value: Maximum value before considering it problematic
        fallback: Value to use if loss is problematic
        name: Name of the loss for logging purposes
        
    Returns:
        Safe loss value that can be backpropagated
    """
    if not torch.isfinite(loss) or loss > max_value:
        # print(f"WARNING: Detected problematic {name} loss value: {loss.item() if torch.isfinite(loss) else 'NaN/Inf'}. Using fallback.")
        return torch.tensor(fallback, device=loss.device, requires_grad=True)
    return loss


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

        # Add numerical stability safeguards
        # Check for NaN values and replace with zeros
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Clip large values to prevent overflow
        x = torch.clamp(x, min=-100, max=100)

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

        # Apply attention mechanism with stability safeguards
        attn_out, _ = self.mha(x, x, x)
        # Add stability clipping
        attn_out = torch.clamp(attn_out, min=-100, max=100)
        x = self.norm1(x + attn_out)

        # Apply feed-forward with stability measures
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        # Add stability clipping
        ffn_out = torch.clamp(ffn_out, min=-100, max=100)
        out = self.norm2(x + ffn_out)

        # Use mean pooling to get fixed-size representation
        out = out.mean(dim=1)  # Shape: (batch_size, embed_dim)
        
        # Final stability check
        out = torch.nan_to_num(out, nan=0.0)
        out = torch.clamp(out, min=-100, max=100)
        
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
    Enhanced with numerical stability safeguards.
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
        
        # Safety check for infinite values - more strict cleaning
        scores = torch.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        
        # Add a small epsilon to prevent numerical issues
        epsilon = 1e-5 # Increased from 1e-6
        
        # Sort targets in descending order with stability
        try:
            # Add extra stability measures and stricter clipping
            # Instead of random noise, use more deterministic approach
            sorted_targets, indices = torch.sort(targets, descending=True, dim=0, stable=True)
            
            # Reorder scores according to target sorting
            ordered_scores = torch.gather(scores, 0, indices)
            
            # Apply very strict clamping to keep values in safe range
            ordered_scores = torch.clamp(ordered_scores, min=epsilon, max=1.0-epsilon)
            
            # Use a higher temperature for more stability
            temperature = 1.0  # Increased from 0.8
            
            # Apply a smoother softmax by scaling inputs to prevent extreme values
            scaled_scores = ordered_scores / temperature
            
            # Scale the scores to a smaller range to prevent overflow
            # Clamp BEFORE log_softmax
            scaled_scores = torch.clamp(scaled_scores, min=-10, max=10)
            
            # Apply log_softmax with stability measures
            scores_softmax = F.log_softmax(scaled_scores, dim=0)
            
            # Very strict clamping to prevent any extreme values
            scores_softmax = torch.clamp(scores_softmax, min=-5, max=0)
            
            # Compute loss with a protective scaling factor
            loss = -torch.mean(scores_softmax) * 0.1  # Scaled down from 0.5 to 0.1
            
            # Final safety check
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.1, device=scores.device, requires_grad=True)
                
            return torch.clamp(loss, min=-1, max=1)
            
        except Exception as e:
            # If anything fails, return a safe fallback value
            print(f"ListMLELoss error: {str(e)}, returning fallback value")
            return torch.tensor(0.1, device=scores.device, requires_grad=True)


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
            # Cap embeddings at 50 dimensions for stability (increased from 25)
            modified_embedding_dims.append(min(50, dim))
        
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
        # SIMPLIFIED: Reduced Toured transformer blocks from 5 to 3
        self.transformer_toured = nn.Sequential(
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout),
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout),
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout=dropout)
        )
        
        # Add a specialized cross-attention mechanism just for toured stage
        self.toured_attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.toured_attention_norm = nn.LayerNorm(transformer_dim)
        
        # Add a special feature enhancement layer for toured prediction
        self.toured_feature_enhancement = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim*2),
            nn.LayerNorm(transformer_dim*2),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),  # Higher dropout for this path
            nn.Linear(transformer_dim*2, transformer_dim)
        )
        
        # Regular transformers for other stages
        self.transformer_applied = nn.Sequential(
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout),
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout) # Add second block
        )
        self.transformer_rented = nn.Sequential(
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout),
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout) # Add second block
        )

        # 5. Prediction heads for each stage
        # SIMPLIFIED: Use standard PredictionHead for Toured stage
        self.toured_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.applied_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.rented_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Stable weight initialization to prevent numerical instability"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use very conservative initialization with much smaller gain
                if 'toured_head' in name or 'transformer_toured' in name:
                    nn.init.xavier_uniform_(module.weight.data, gain=0.1)  # Reduced from 0.3
                else:
                    nn.init.xavier_uniform_(module.weight.data, gain=0.1)  # Reduced from 0.2
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
                
            elif isinstance(module, nn.Embedding):
                # Very careful initialization for embeddings
                nn.init.normal_(module.weight.data, mean=0.0, std=0.01)  # Reduced std
                
            elif isinstance(module, nn.MultiheadAttention):
                # Special initialization for attention mechanisms with small values
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=0.05)  # Reduced gain
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=0.05)  # Reduced gain
                
                # Initialize any bias terms
                for param_name, param in module.named_parameters():
                    if 'bias' in param_name:
                        nn.init.zeros_(param)

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
        
        # Apply a simpler, more stable residual approach instead of full cross-attention
        # Clamp the input values to prevent extreme values
        toured_features_safe = torch.clamp(toured_features, min=-10, max=10)
        # Use LayerNorm to help stabilize the features
        toured_features = self.toured_attention_norm(toured_features_safe)

        # Then continue with feature enhancement
        toured_features = self.toured_feature_enhancement(toured_features)

        # Ensure no NaN/Inf values in features before prediction
        toured_features = torch.nan_to_num(toured_features, nan=0.0)
        toured_features = torch.clamp(toured_features, min=-10, max=10)
        
        # Apply simpler toured_head
        toured_pred = self.toured_head(toured_features)
        
        # Strong clamping for predictions to ensure they stay in valid sigmoid range
        toured_pred = torch.clamp(toured_pred, min=1e-6, max=1-1e-6)

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


def find_difficult_toured_examples(train_loader, model, fraction=0.2, device=None):
    """
    Find difficult examples for toured predictions based on incorrect predictions
    or high loss values. Used for targeted learning.
    
    Args:
        train_loader: DataLoader for training data
        model: The model to evaluate
        fraction: Fraction of difficult examples to collect
        device: Device to use (defaults to model's device)
        
    Returns:
        difficult_cat, difficult_num, difficult_tour: Tensors of selected difficult examples
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    all_cat = []
    all_num = []
    all_tour = []
    all_loss = []
    
    # Sample a subset of batches to find difficult examples
    with torch.no_grad():
        # Only check up to 20 batches to save time
        for i, batch in enumerate(train_loader):
            if i >= 20:  # Limit to first 20 batches to save time
                break
                
            cat_in = batch[0].to(device)
            num_in = batch[1].to(device)
            tour_labels = batch[2].to(device)
            
            # Get predictions
            outputs = model(cat_in, num_in)
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                tour_pred = outputs[0]
            else:
                tour_pred = outputs
            
            # Calculate loss for each example
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
            losses = criterion(tour_pred, tour_labels)
            
            # Convert to numpy for easier handling
            cat_cpu = cat_in.cpu()
            num_cpu = num_in.cpu()
            tour_cpu = tour_labels.cpu()
            loss_cpu = losses.cpu()
            
            # Store data
            all_cat.append(cat_cpu)
            all_num.append(num_cpu)
            all_tour.append(tour_cpu)
            all_loss.append(loss_cpu)
    
    if not all_cat:
        return None, None, None
        
    # Combine all batches
    all_cat = torch.cat(all_cat, dim=0)
    all_num = torch.cat(all_num, dim=0)
    all_tour = torch.cat(all_tour, dim=0)
    all_loss = torch.cat(all_loss, dim=0)
    
    # Find the indices of the most difficult examples (highest loss)
    _, difficult_indices = torch.topk(all_loss.squeeze(), k=int(fraction * len(all_loss)))
    
    # Select the difficult examples
    difficult_cat = all_cat[difficult_indices].to(device)
    difficult_num = all_num[difficult_indices].to(device)
    difficult_tour = all_tour[difficult_indices].to(device)
    
    return difficult_cat, difficult_num, difficult_tour


# Training function for cascaded model with ranking loss
def train_cascaded_model(model,
                         train_loader,
                         valid_loader,
                         optimizer,
                         scheduler,
                         num_epochs=30,
                         toured_weight=4.0,  # Increased from 1.0
                         applied_weight=1.0,
                         rented_weight=2.0,
                         ranking_weight=0.2,
                         device='cuda',
                         early_stopping_patience=5,
                         model_save_path='best_model.pt',
                         mixed_precision=True,
                         gradient_accumulation_steps=1,
                         verbose=True,
                         epoch_csv_path=None,
                         focus_on_difficult_examples=True,
                         max_grad_norm=0.5):  # New parameter for strict gradient clipping
    """
    Training loop for cascaded model with ranking loss and specialized toured training
    with enhanced numerical stability.
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
            'toured_p50', 'applied_p50', 'rented_p50',
            'toured_pmax', 'applied_pmax', 'rented_pmax'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device == 'cuda') else None

    # Loss functions
    # Use Focal Loss instead of BCE
    # Lower alpha for majority class (Toured), higher for minority (Applied/Rented)
    toured_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    applied_criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')
    rented_criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')

    # Ranking loss
    ranking_criterion = ListMLELoss()

    # Initialize weight variables with default values
    current_toured_weight = toured_weight
    current_applied_weight = applied_weight  
    current_rented_weight = rented_weight

    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # NEW: Warm-up settings
    warm_up_epochs = 5
    initial_lr = 1e-7 # Start very low
    target_lr = optimizer.param_groups[0]['lr'] # Get target LR from optimizer

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
        # NEW: Warm-up settings
        if epoch < warm_up_epochs:
            # Apply linear warm-up
            lr = initial_lr + (target_lr - initial_lr) * (epoch / warm_up_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if verbose:
                print(f"\nWARM-UP PHASE (Epoch {epoch+1}/{warm_up_epochs}): Set LR to {lr:.6f}")
                
            # During warm-up, still focus more on toured, but less aggressively
            current_toured_weight = toured_weight * 1.0 # Full weight during warm-up
            current_applied_weight = applied_weight * 0.5 # Reduced weight during warm-up
            current_rented_weight = rented_weight * 0.5 # Reduced weight during warm-up
            use_ranking_loss = False # No ranking during warm-up
            focus_this_epoch = False # No difficult example focus during warm-up
            
        else: # After warm-up
            # Ensure LR is at target (scheduler might change it later)
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
                
            current_toured_weight = toured_weight
            current_applied_weight = applied_weight
            current_rented_weight = rented_weight
            use_ranking_loss = True # Enable ranking loss after warm-up
            focus_this_epoch = focus_on_difficult_examples and epoch % 3 == 0 # Check flag and epoch number
            
            if verbose and epoch == warm_up_epochs:
                 print(f"\nWARM-UP COMPLETE. Using full LR: {target_lr:.6f} and loss weights.")
                 if use_ranking_loss:
                     print("   - Ranking loss enabled.")

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

        # NEW: Focus on difficult toured examples every few epochs after warm-up
        # Only run if focus_this_epoch is True
        if focus_this_epoch:
            if verbose:
                print("\nFinding difficult toured examples for focused training...") # Added print
                
            difficult_cat, difficult_num, difficult_tour = find_difficult_toured_examples(
                train_loader, model
            )
            
            if difficult_cat is not None and len(difficult_cat) > 0:
                if verbose:
                    print(f"Found {len(difficult_cat)} difficult examples. Performing focused training...")
                
                # Train for several iterations on difficult examples
                for _ in range(3):  # 3 iterations of focused training
                    optimizer.zero_grad()
                    
                    if scaler is not None:
                        # Mixed precision
                        with torch.cuda.amp.autocast():
                            outputs = model(difficult_cat, difficult_num)
                            if isinstance(outputs, tuple) and len(outputs) >= 3:
                                # Extract only toured predictions (first output)
                                tour_pred = outputs[0]
                            else:
                                tour_pred = outputs
                            
                            # Focus exclusively on toured loss with higher weight
                            tour_losses = toured_criterion(tour_pred, difficult_tour)
                            tour_losses = torch.clamp(tour_losses, 0, 5)
                            tour_loss = tour_losses.mean()
                            
                            # Apply extra weight for difficult examples
                            focused_loss = 3.0 * toured_weight * tour_loss
                            
                        scaler.scale(focused_loss).backward()
                        scaler.unscale_(optimizer)
                        # Use the passed max_grad_norm parameter instead of hardcoded value
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # More strict gradient clipping
                        scaler.step(optimizer)
                    else:
                        # No mixed precision
                        outputs = model(difficult_cat, difficult_num)
                        if isinstance(outputs, tuple) and len(outputs) >= 3:
                            tour_pred = outputs[0]
                        else:
                            tour_pred = outputs
                        
                        # Focus exclusively on toured loss
                        tour_losses = toured_criterion(tour_pred, difficult_tour)
                        tour_losses = torch.clamp(tour_losses, 0, 5)
                        tour_loss = tour_losses.mean()
                        
                        # Apply extra weight for difficult examples
                        focused_loss = 3.0 * toured_weight * tour_loss
                        
                        focused_loss.backward()
                        # Use the passed max_grad_norm parameter
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # More strict gradient clipping
                        optimizer.step()
                        optimizer.zero_grad()
                
                if verbose:
                    print("Completed focused training phase.") # Added print
        
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
                        # SIMPLIFIED LOSS CALCULATION
                        toured_loss = toured_criterion(toured_pred, toured_labels)
                        # toured_losses = torch.clamp(toured_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                        # toured_losses = toured_losses * 0.8 + 0.05  # More aggressive smoothing
                        # toured_loss = toured_losses.mean()
                        # toured_loss = safe_loss(toured_loss, max_value=2.0, fallback=0.7, name="Toured") # Reduced max_value from 3.0

                        # For applied loss, create the tour_gt_mask
                        # toured_gt_mask = (toured_labels == 1).float() * 1.5 + 0.5  # Weight 2.0 for true, 0.5 for false
                        applied_loss = applied_criterion(applied_pred, applied_labels)
                        # applied_losses = torch.clamp(applied_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                        # applied_losses = applied_losses * 0.8 + 0.05  # More aggressive smoothing
                        # applied_loss = (applied_losses * toured_gt_mask).mean()
                        # applied_loss = safe_loss(applied_loss, max_value=2.0, fallback=0.7, name="Applied") # Reduced max_value from 3.0

                        # For rented loss, create the applied_gt_mask
                        # applied_gt_mask = (applied_labels == 1).float() * 1.5 + 0.5  # Weight 2.0 for true, 0.5 for false
                        rented_loss = rented_criterion(rented_pred, rented_labels)
                        # rented_losses = torch.clamp(rented_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                        # rented_losses = rented_losses * 0.8 + 0.05  # More aggressive smoothing
                        # rented_loss = (rented_losses * applied_gt_mask).mean()
                        # rented_loss = safe_loss(rented_loss, max_value=2.0, fallback=0.7, name="Rented") # Reduced max_value from 3.0

                        # Calculate bce_loss with the current weights
                        bce_loss = current_toured_weight * toured_loss + \
                                   current_applied_weight * applied_loss + \
                                   current_rented_weight * rented_loss

                        # Final safety check on combined loss
                        # bce_loss = safe_loss(bce_loss, max_value=2.0, fallback=1.0, name="BCE Combined")

                        # Add Ranking Loss - only after warm-up and with enough batch size
                        ranking_loss_value = torch.tensor(0.0, device=device) # Default to 0
                        if use_ranking_loss and categorical_inputs.size(0) >= 10:
                            try:
                                toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                                applied_ranking_loss = ranking_criterion(applied_pred.squeeze(),
                                                                         applied_labels.squeeze())
                                rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                                # Apply safe_loss to individual ranking losses
                                # toured_ranking_loss = safe_loss(toured_ranking_loss, max_value=1.0, fallback=0.1, name="Toured Ranking")
                                # applied_ranking_loss = safe_loss(applied_ranking_loss, max_value=1.0, fallback=0.1, name="Applied Ranking")
                                # rented_ranking_loss = safe_loss(rented_ranking_loss, max_value=1.0, fallback=0.1, name="Rented Ranking")

                                ranking_loss_value = current_toured_weight * toured_ranking_loss + \
                                               current_applied_weight * applied_ranking_loss + \
                                               current_rented_weight * rented_ranking_loss
                               
                                # Apply safe_loss to combined ranking loss
                                # ranking_loss_value = safe_loss(ranking_loss_value, max_value=1.5, fallback=0.2, name="Ranking Combined")

                                # Combine BCE and ranking loss
                                loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss_value
                            except Exception as e:
                                # If ranking loss fails, just use BCE only
                                if verbose:
                                    print(f"Ranking loss calculation failed: {str(e)}, using BCE only for this batch")
                                loss = bce_loss
                        else:
                            # Use only BCE during warm-up, if batch is too small, or if ranking is disabled for the epoch
                            loss = bce_loss

                        # Final safety check on loss value - more conservative clamping
                        if not torch.isfinite(loss) or loss > 5:
                            if verbose:
                                print(f"WARNING: Non-finite loss detected: {loss}. Using fallback value.")
                                # Print components for debugging
                                print(f"  Components: BCE={bce_loss.item() if torch.isfinite(bce_loss) else 'NaN/Inf'}",
                                      f"Rank={ranking_loss_value.item() if torch.isfinite(ranking_loss_value) else 'NaN/Inf'}",
                                      f"UseRank={use_ranking_loss}")
                            loss = torch.tensor(1.0, device=device, requires_grad=True)  # Safer fallback value

                        # Final safety check on total loss
                        # loss = safe_loss(loss, max_value=2.0, fallback=1.0, name="Total Loss")

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
                        # Use the passed max_grad_norm parameter instead of hardcoded value
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # More strict gradient clipping
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
                    # SIMPLIFIED LOSS
                    toured_loss = toured_criterion(toured_pred, toured_labels)
                    # toured_losses = torch.clamp(toured_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                    # toured_losses = toured_losses * 0.8 + 0.05  # More aggressive smoothing
                    # toured_loss = toured_losses.mean()
                    # toured_loss = safe_loss(toured_loss, max_value=2.0, fallback=0.7, name="Toured-NM") # Reduced max_value from 3.0

                    # For applied loss, create the tour_gt_mask
                    # toured_gt_mask = (toured_labels == 1).float() * 1.5 + 0.5  # Weight 2.0 for true, 0.5 for false
                    applied_loss = applied_criterion(applied_pred, applied_labels)
                    # applied_losses = torch.clamp(applied_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                    # applied_losses = applied_losses * 0.8 + 0.05  # More aggressive smoothing 
                    # applied_loss = (applied_losses * toured_gt_mask).mean()
                    # applied_loss = safe_loss(applied_loss, max_value=2.0, fallback=0.7, name="Applied-NM") # Reduced max_value from 3.0

                    # For rented loss, create the applied_gt_mask
                    # applied_gt_mask = (applied_labels == 1).float() * 1.5 + 0.5  # Weight 2.0 for true, 0.5 for false
                    rented_loss = rented_criterion(rented_pred, rented_labels)
                    # rented_losses = torch.clamp(rented_losses, 0, 1.0)  # Reduced from 3.0 to 1.0
                    # rented_losses = rented_losses * 0.8 + 0.05  # More aggressive smoothing
                    # rented_loss = (rented_losses * applied_gt_mask).mean()
                    # rented_loss = safe_loss(rented_loss, max_value=2.0, fallback=0.7, name="Rented-NM") # Reduced max_value from 3.0

                    # Calculate bce_loss with the current weights
                    bce_loss = current_toured_weight * toured_loss + \
                               current_applied_weight * applied_loss + \
                               current_rented_weight * rented_loss

                    # Final safety check on combined loss
                    # bce_loss = safe_loss(bce_loss, max_value=2.0, fallback=1.0, name="BCE Combined")

                    # Add Ranking Loss - only after warm-up period and if enabled for epoch
                    ranking_loss_value = torch.tensor(0.0, device=device) # Default to 0
                    if use_ranking_loss and categorical_inputs.size(0) >= 10:
                        try:
                            toured_ranking_loss = ranking_criterion(toured_pred.squeeze(), toured_labels.squeeze())
                            applied_ranking_loss = ranking_criterion(applied_pred.squeeze(), applied_labels.squeeze())
                            rented_ranking_loss = ranking_criterion(rented_pred.squeeze(), rented_labels.squeeze())

                            # Apply safe_loss to individual ranking losses
                            # toured_ranking_loss = safe_loss(toured_ranking_loss, max_value=1.0, fallback=0.1, name="Toured Ranking-NM")
                            # applied_ranking_loss = safe_loss(applied_ranking_loss, max_value=1.0, fallback=0.1, name="Applied Ranking-NM")
                            # rented_ranking_loss = safe_loss(rented_ranking_loss, max_value=1.0, fallback=0.1, name="Rented Ranking-NM")

                            ranking_loss_value = current_toured_weight * toured_ranking_loss + \
                                           current_applied_weight * applied_ranking_loss + \
                                           current_rented_weight * rented_ranking_loss
                           
                            # Apply safe_loss to combined ranking loss
                            # ranking_loss_value = safe_loss(ranking_loss_value, max_value=1.5, fallback=0.2, name="Ranking Combined-NM")

                            # Combine BCE and ranking loss
                            loss = (1.0 - ranking_weight) * bce_loss + ranking_weight * ranking_loss_value
                        except Exception as e:
                            if verbose:
                                print(f"Ranking loss calculation failed: {str(e)}, using BCE only for this batch")
                            loss = bce_loss
                    else:
                        # Use only BCE during warm-up or if batch is too small
                        loss = bce_loss

                    # Final safety check on total loss
                    # loss = safe_loss(loss, max_value=2.0, fallback=1.0, name="Total Loss-NM")

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

                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Use the passed max_grad_norm parameter instead of hardcoded value
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # More strict gradient clipping
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

                    # Simple validation loss using the same FocalLoss functions
                    toured_loss = toured_criterion(toured_pred, toured_labels)
                    applied_loss = applied_criterion(applied_pred, applied_labels)
                    rented_loss = rented_criterion(rented_pred, rented_labels)
                    
                    batch_loss = toured_loss + applied_loss + rented_loss # Simple sum for validation loss metric

                    # Stricter clipping for batch loss
                    if not torch.isfinite(batch_loss) or batch_loss > 10: # Reduced max validation loss check
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
            
            # Precision@max (using all examples)
            k_max = len(all_toured_true)
            toured_pmax = precision_at_k(all_toured_true.flatten(), all_toured_preds.flatten(), k_max)
            applied_pmax = precision_at_k(all_applied_true.flatten(), all_applied_preds.flatten(), k_max)
            rented_pmax = precision_at_k(all_rented_true.flatten(), all_rented_preds.flatten(), k_max)
            
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
                history['toured_pmax'] = []
                history['applied_pmax'] = []
                history['rented_pmax'] = []
                
            history['toured_auc'].append(toured_auc)
            history['applied_auc'].append(applied_auc)
            history['rented_auc'].append(rented_auc)
            history['toured_apr'].append(toured_apr)
            history['applied_apr'].append(applied_apr)
            history['rented_apr'].append(rented_apr)
            history['toured_p50'].append(toured_p50)
            history['applied_p50'].append(applied_p50)
            history['rented_p50'].append(rented_p50)
            history['toured_pmax'].append(toured_pmax)
            history['applied_pmax'].append(applied_pmax)
            history['rented_pmax'].append(rented_pmax)
            
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
                    'rented_p50': rented_p50,
                    'toured_pmax': toured_pmax,
                    'applied_pmax': applied_pmax,
                    'rented_pmax': rented_pmax
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
                print(f"\nValidation Metrics:")
                print(f"  Toured:  AUC={toured_auc:.4f}  APR={toured_apr:.4f}  P@50={toured_p50:.4f}  P@max={toured_pmax:.4f}")
                print(f"  Applied: AUC={applied_auc:.4f}  APR={applied_apr:.4f}  P@50={applied_p50:.4f}  P@max={applied_pmax:.4f}")
                print(f"  Rented:  AUC={rented_auc:.4f}  APR={rented_apr:.4f}  P@50={rented_p50:.4f}  P@max={rented_pmax:.4f}")
                print(f"  Max predictions (k_max): {k_max}")
            print("-" * 60)

        # Step LR scheduler *after* warm-up phase
        if epoch >= warm_up_epochs:
            scheduler.step() # Step the main scheduler (e.g., CosineAnnealingLR)
        # else: # During warm-up, LR is set manually
            # pass 

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

# NEW: Focal Loss Implementation
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is used to address dataset imbalance. It forces training to focus learning
    on hard examples and down-weights the loss value contributed by easy examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """Focal Loss

        Args:
            alpha (float, optional): Weighting factor for the positive class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed.
                Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass

        Args:
            inputs (torch.Tensor): Predicted probabilities, shape [N, 1] or [N].
            targets (torch.Tensor): Ground truth labels, shape [N, 1] or [N].

        Returns:
            torch.Tensor: Calculated focal loss.
        """
        # Ensure inputs and targets are same shape
        if inputs.shape != targets.shape:
            targets = targets.view_as(inputs)
            
        # Ensure inputs are probabilities (0-1)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Computes pt = p if targets=1, pt = 1-p if targets=0
        
        # Calculate alpha weight for each sample
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate Focal loss
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:  # 'none'
            return F_loss