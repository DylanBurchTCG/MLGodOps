import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimplifiedTourModel(nn.Module):
    def __init__(self,
                 categorical_dims,
                 embedding_dims,
                 numerical_dim,
                 hidden_units=[256, 128, 64],
                 dropout=0.3,
                 use_batch_norm=True):
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim)
            for dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])

        # Calculate total input dimension
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + numerical_dim

        # Simplified feature attention - just one layer of attention instead of multiple
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Build MLP layers with batch norm and dropout
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_units:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.batch_norms.append(nn.Identity())

            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final prediction layer
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_units[-1], 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, categorical_inputs, numerical_inputs):
        # Process categorical features
        embeddings = []
        for i, embedding_layer in enumerate(self.embeddings):
            embeddings.append(embedding_layer(categorical_inputs[:, i]))

        # Concatenate embeddings horizontally
        if embeddings:
            x_cat = torch.cat(embeddings, dim=1)
            # Concatenate with numerical features
            if numerical_inputs is not None and numerical_inputs.size(1) > 0:
                x = torch.cat([x_cat, numerical_inputs], dim=1)
            else:
                x = x_cat
        else:
            # Only numerical features
            x = numerical_inputs

        # Apply feature attention
        x = self.feature_attention(x)

        # Forward through MLP network
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.gelu(x)  # Use GELU for smoother gradients
            x = self.batch_norms[i](x)
            x = self.dropouts[i](x)

        # Final prediction
        predictions = self.prediction_head(x)

        # Return predictions and None for attention weights to maintain interface compatibility
        return predictions, None