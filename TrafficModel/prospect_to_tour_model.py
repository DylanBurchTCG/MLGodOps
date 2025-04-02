import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import joblib

class TourPredictionModel(nn.Module):
    def __init__(self, 
                 categorical_dims,
                 embedding_dims,
                 numerical_dim,
                 hidden_units=[512, 256, 128],
                 dropout=0.2,  # Reduced dropout for moderately imbalanced data
                 use_batch_norm=True,
                 toured_rate=0.3712):  # Add toured rate parameter
        super().__init__()
        
        self.toured_rate = toured_rate  # Store the class distribution
        
        # Embedding layer for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) 
            for dim, emb_dim in zip(categorical_dims, embedding_dims)
        ])
        
        # Calculate total input dimension
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + numerical_dim
        
        # Feature enhancement layer with attention
        self.feature_enhancement = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Multi-head attention for feature importance
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(input_dim)
        
        # Multi-layer network with residual connections and batch normalization
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_units):
            # Main layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.batch_norms.append(nn.Identity())
                
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
            
            # Residual connection (if dimensions match)
            if prev_dim == hidden_dim:
                self.residual_layers.append(nn.Identity())
            else:
                self.residual_layers.append(nn.Linear(prev_dim, hidden_dim))
                
            prev_dim = hidden_dim
        
        # Final prediction layer with adaptive threshold
        self.toured_head = nn.Sequential(
            nn.Linear(hidden_units[-1], 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
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
            
        # Feature enhancement
        x = self.feature_enhancement(x)
        
        # Apply multi-head attention
        x = x.unsqueeze(1)  # Add sequence dimension
        x_attended, attention_weights = self.feature_attention(x, x, x)
        x = x_attended.squeeze(1)
        x = self.attention_norm(x)
        
        # Forward through deep network with residual connections
        for i in range(len(self.layers)):
            # Main path
            z = self.layers[i](x)
            z = F.gelu(z)  # Using GELU for better gradient flow
            z = self.batch_norms[i](z)
            z = self.dropouts[i](z)
            
            # Residual connection
            res = self.residual_layers[i](x)
            x = z + res
        
        # Final prediction
        toured_pred = self.toured_head(x)
        
        return toured_pred, attention_weights
