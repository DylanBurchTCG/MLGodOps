import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    """
    Handles embedding of categorical features and processing of numerical features
    """

    def __init__(self, categorical_dims, embedding_dims, numerical_dim):
        super().__init__()
        # Replace embedding layers with linear layers for target-encoded features
        self.categorical_layers = nn.ModuleList([
            nn.Linear(1, emb_dim)
            for cat_dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        self.numerical_bn = nn.BatchNorm1d(numerical_dim) if numerical_dim > 0 else None

    def forward(self, categorical_inputs, numerical_inputs):
        embedded_features = []

        # Process categorical features
        if len(self.categorical_layers) > 0:
            for i, layer in enumerate(self.categorical_layers):
                # Reshape to have proper dimensions for linear layer
                cat_feature = categorical_inputs[:, i].view(-1, 1)
                embedded_features.append(layer(cat_feature))

        # Process numerical features
        if self.numerical_bn is not None:
            numerical_features = self.numerical_bn(numerical_inputs)
        else:
            numerical_features = numerical_inputs

        # Combine features
        all_features = []

        # Add categorical features if we have any
        if embedded_features:
            # Concatenate all embedded features
            categorical_output = torch.cat(embedded_features, dim=1)
            all_features.append(categorical_output)

        # Add numerical features if we have any
        if numerical_features.size(1) > 0:
            all_features.append(numerical_features)

        # Concatenate everything
        if all_features:
            return torch.cat(all_features, dim=1)
        else:
            # This should never happen if we have either categorical or numerical features
            return torch.zeros((categorical_inputs.size(0), 0), device=categorical_inputs.device)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for tabular data
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape for attention: [batch_size, seq_len, embed_dim]
        batch_size = x.size(0)
        # Calculate number of "tokens" based on the input size and embedding dimension
        seq_len = x.size(1) // self.embed_dim

        # Make sure we have a multiple of embed_dim
        if seq_len * self.embed_dim != x.size(1):
            # Add padding or adjust x if needed
            padding_size = seq_len * self.embed_dim
            if padding_size < x.size(1):
                seq_len += 1
                padding_size = seq_len * self.embed_dim

            padding = torch.zeros(batch_size, padding_size - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Reshape to [batch_size, seq_len, embed_dim]
        x_reshaped = x.view(batch_size, seq_len, self.embed_dim)

        # Apply self-attention
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)

        # Add residual connection and normalize
        output = self.norm(x_reshaped + attn_output)

        # Reshape back to original dimensions
        output = output.reshape(batch_size, -1)

        # If we added padding, remove it now
        if output.size(1) > x.size(1):
            output = output[:, :x.size(1)]

        return output


class TabularTransformerBlock(nn.Module):
    """
    Transformer block adapted for tabular data
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
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
        # Make sure x is the right dimension for the attention layer
        batch_size = x.size(0)

        # Check if x needs reshaping
        if x.dim() == 2 and x.size(1) != self.embed_dim:
            # Assuming x is [batch_size, features]
            # We'll reshape it to [batch_size, seq_len, embed_dim]
            seq_len = x.size(1) // self.embed_dim

            # Ensure we have a valid reshape
            if seq_len * self.embed_dim != x.size(1):
                # Need to pad to make divisible
                padding_needed = self.embed_dim - (x.size(1) % self.embed_dim)
                if padding_needed < self.embed_dim:
                    padding = torch.zeros(batch_size, padding_needed, device=x.device)
                    x = torch.cat([x, padding], dim=1)
                    seq_len = x.size(1) // self.embed_dim

            # Reshape to [batch_size, seq_len, embed_dim]
            x = x.view(batch_size, seq_len, self.embed_dim)

        # Self-attention with residual connection and normalization
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output) if x.dim() == 3 else self.norm1(x)

        # Feed-forward network with residual connection and normalization
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        output = self.norm2(x + ffn_output)

        # Reshape back to original dimensions if needed
        if output.dim() == 3 and output.size(2) == self.embed_dim:
            output = output.reshape(batch_size, -1)

        return output


class PredictionHead(nn.Module):
    """
    Stage-specific prediction head
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)
        self.attention_gates = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply feature-wise attention gates
        attention_weights = self.attention_gates(x)
        attended_features = x * attention_weights

        # Make prediction
        return self.layers(attended_features)


class LeadFunnelModel(nn.Module):
    """
    Multi-stage model for lead funnel prediction
    """

    def __init__(self, categorical_dims, embedding_dims, numerical_dim,
                 transformer_dim, num_heads, num_transformer_blocks,
                 ff_dim, head_hidden_dims, dropout=0.2):
        super().__init__()

        # Feature embedding
        self.embedding_layer = EmbeddingLayer(categorical_dims, embedding_dims, numerical_dim)

        # Compute total embedding dimension
        total_embedding_dim = sum(embedding_dims) + numerical_dim

        # Projection to transformer dimension
        self.projection = nn.Linear(total_embedding_dim, transformer_dim)

        # TabTransformer blocks
        self.transformer_blocks = nn.ModuleList([
            TabularTransformerBlock(transformer_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # Stage-specific heads
        self.toured_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.applied_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.rented_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)

    def forward(self, categorical_inputs, numerical_inputs):
        # Embed features
        x = self.embedding_layer(categorical_inputs, numerical_inputs)

        # Debug dimensions
        batch_size = x.size(0)

        # Project to transformer dimension
        x = self.projection(x)

        # Prepare input for transformer blocks
        # Each transformer block will handle the reshaping internally

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply prediction heads
        toured_pred = self.toured_head(x)
        applied_pred = self.applied_head(x)
        rented_pred = self.rented_head(x)

        return toured_pred, applied_pred, rented_pred

    def predict_toured(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.toured_head(x)

    def predict_applied(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.applied_head(x)

    def predict_rented(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.rented_head(x)