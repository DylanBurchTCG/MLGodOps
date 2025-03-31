import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    """
    Handles embedding of categorical features and processing of numerical features
    """

    def __init__(self, categorical_dims, embedding_dims, numerical_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim)
            for cat_dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        self.numerical_bn = nn.BatchNorm1d(numerical_dim) if numerical_dim > 0 else None

    def forward(self, categorical_inputs, numerical_inputs):
        embedded = []
        if len(self.embeddings) > 0:
            for i, emb in enumerate(self.embeddings):
                embedded.append(emb(categorical_inputs[:, i]))
            embedded = torch.cat(embedded, dim=1)

        if self.numerical_bn is not None:
            numerical = self.numerical_bn(numerical_inputs)
            combined = torch.cat([embedded, numerical], dim=1) if embedded else numerical
        else:
            combined = embedded

        return combined


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for tabular data
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Reshape for attention: [batch_size, seq_len, embed_dim]
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1, 1)  # Treat each feature as a "token"

        # Apply self-attention
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)

        # Add residual connection and normalize
        output = self.norm(x_reshaped + attn_output)

        # Reshape back to original dimensions
        output = output.reshape(batch_size, -1)
        return output


class TabularTransformerBlock(nn.Module):
    """
    Transformer block adapted for tabular data
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout(ffn_output)

        # Add residual connection and normalize
        output = self.norm(attn_output + ffn_output)
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
        self.tour_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.apply_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)
        self.rent_head = PredictionHead(transformer_dim, head_hidden_dims, dropout)

    def forward(self, categorical_inputs, numerical_inputs):
        # Embed features
        x = self.embedding_layer(categorical_inputs, numerical_inputs)

        # Project to transformer dimension
        x = self.projection(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply prediction heads
        tour_pred = self.tour_head(x)
        apply_pred = self.apply_head(x)
        rent_pred = self.rent_head(x)

        return tour_pred, apply_pred, rent_pred

    def predict_tour(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.tour_head(x)

    def predict_apply(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.apply_head(x)

    def predict_rent(self, categorical_inputs, numerical_inputs):
        x = self.embedding_layer(categorical_inputs, numerical_inputs)
        x = self.projection(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.rent_head(x)