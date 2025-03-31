import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # If file not found, try using a sample of the data for testing
        print(f"Could not find {file_path}, creating sample data")

        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 10000
        n_features = 50

        data = np.random.randn(n_samples, n_features)

        # Create a target variable with some class imbalance
        target = np.zeros(n_samples)
        for i in range(n_samples):
            prob = 0.3 + 0.4 * np.sin(data[i, 0]) + 0.3 * np.cos(data[i, 1])
            target[i] = np.random.binomial(1, prob)

        # Convert to DataFrame
        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)

        # Add some categorical variables
        df['cat_var_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
        df['cat_var_2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)

        # Add the target variable
        df['qualified'] = target.astype(int)

        print("Created sample data with shape:", df.shape)

        return df

    # Create the 'qualified' column based on conditions if it doesn't exist
    if 'qualified' not in df.columns:
        if all(col in df.columns for col in ['TOTAL_APPLIED', 'TOTAL_APPOINTMENTS_COMPLETED', 'TOTAL_RENTED']):
            df['qualified'] = ((df['TOTAL_APPLIED'] == 1) |
                               (df['TOTAL_APPOINTMENTS_COMPLETED'] == 1) |
                               (df['TOTAL_RENTED'] == 1)).astype(int)
        else:
            raise ValueError("Cannot create 'qualified' column, required columns are missing")

    # Columns to drop if they exist
    columns_to_drop = [
        'TOTAL_APPLIED',
        'TOTAL_APPOINTMENTS_COMPLETED',
        'TOTAL_RENTED',
        'TOTAL_APPOINTMENTS_SCHEDULED',
        'CLIENT_ID',
        'CLIENT_PERSON_ID',
        'EXTRACT_DATE',
        'RECD_P_ID',
        'FNAM_FNAM',
        'MNAM_MNAM',
        'PFXT_PFXT',
        'SNAM_SNAM',
        'SFXT_SFXT',
        'PERSTYPE',
        'BDAT_BDAT',
    ]

    # Drop columns that exist
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    print(f"Data shape after preprocessing: {df.shape}")

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: Dataset contains {nan_count} NaN values. Will handle with imputation.")

    return df


# Feature engineering function to enhance the dataset
def engineer_features(df):
    """Add engineered features to improve model performance"""
    print("Adding engineered features...")

    # Copy to avoid modifying the original
    df_new = df.copy()

    # Find numeric columns
    numeric_cols = df_new.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # If we have COMBINED_AGE, create age groups
    if 'COMBINED_AGE' in numeric_cols:
        # Age buckets
        df_new['AGE_GROUP'] = pd.cut(
            df_new['COMBINED_AGE'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )

    # If we have specific columns, create interaction terms
    interaction_candidates = [
        'HEALTH_WELL_BEING', 'TECHNOLOGY_ADOPTION', 'POLITICAL_CODE',
        'EMAIL_FLAG', 'GNDR_GNDR'
    ]

    # Create interaction features for pairs of columns that exist
    existing_cols = [col for col in interaction_candidates if col in df_new.columns]

    # If we have at least 2 columns for interactions
    if len(existing_cols) >= 2:
        for i in range(len(existing_cols)):
            for j in range(i + 1, len(existing_cols)):
                col1 = existing_cols[i]
                col2 = existing_cols[j]

                # Create interaction feature if both are numeric
                if col1 in numeric_cols and col2 in numeric_cols:
                    df_new[f'{col1}_{col2}_INTERACT'] = df_new[col1] * df_new[col2]

    return df_new


# Function to split data and encode categorical features with NaN handling
def prepare_data(df, oversample=True):
    print("Preparing data for modeling...")

    # Apply feature engineering
    df = engineer_features(df)

    # Separate features and target
    X = df.drop('qualified', axis=1)
    y = df['qualified']

    # Check class imbalance
    class_counts = np.bincount(y)
    print(f"Class distribution - 0: {class_counts[0]}, 1: {class_counts[1]}")

    # Check for NaN values in each column
    columns_with_nan = X.columns[X.isna().any()].tolist()
    if columns_with_nan:
        print(f"Found {len(columns_with_nan)} columns with NaN values: {columns_with_nan[:5]}...")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Add other columns that should be treated as categorical
    categorical_candidates = [col for col in X.columns if col not in categorical_features]
    for col in categorical_candidates:
        # If a column has fewer than 15 unique values, treat it as categorical
        if X[col].nunique() < 15:
            categorical_features.append(col)

    numerical_features = [col for col in X.columns if col not in categorical_features]

    print(f"Number of categorical features: {len(categorical_features)}")
    print(f"Number of numerical features: {len(numerical_features)}")

    # Build preprocessing pipelines with imputation for NaN handling
    # For numerical features: use median imputation and robust scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # For categorical features: use most frequent imputation and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Process the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        categorical_features)

    # Combine numerical and one-hot encoded feature names
    feature_names = np.concatenate([numerical_features, onehot_feature_names])

    # Apply SMOTE for oversampling the minority class if requested
    if oversample:
        try:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            print(f"After SMOTE - Shape: {X_train_processed.shape}, Class distribution: {np.bincount(y_train)}")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            print("Continuing without SMOTE...")

    # Convert to DataFrames
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

    print(f"Train data shape: {X_train_processed_df.shape}")
    print(f"Test data shape: {X_test_processed_df.shape}")

    # Check if any NaN values remain in processed data
    train_nan_count = np.isnan(X_train_processed).sum()
    test_nan_count = np.isnan(X_test_processed).sum()

    if train_nan_count > 0 or test_nan_count > 0:
        print(f"WARNING: Processed data still contains NaN values. Train: {train_nan_count}, Test: {test_nan_count}")
        # Replace any remaining NaNs with 0 as a last resort
        X_train_processed = np.nan_to_num(X_train_processed)
        X_test_processed = np.nan_to_num(X_test_processed)
        print("Replaced remaining NaN values with 0.")

        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return X_train_processed_df, X_test_processed_df, y_train, y_test, numerical_features, categorical_features, preprocessor


# PyTorch Dataset class for neural network
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Attention-based Self-Attention Block for tabular data
class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=64, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads

        # Multi-head attention projections
        self.q_linear = nn.Linear(input_dim, attention_dim)
        self.k_linear = nn.Linear(input_dim, attention_dim)
        self.v_linear = nn.Linear(input_dim, attention_dim)

        self.out_proj = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply layer normalization first (pre-norm)
        residual = x
        x = self.layer_norm(x)

        batch_size = x.size(0)

        # Reshape for multi-head attention:
        # Add a sequence dimension of 1 before processing if it doesn't exist
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, feature_dim]

        # Project inputs to queries, keys, and values
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_dim)

        # Project back to input dimension
        output = self.out_proj(context)
        output = self.dropout(output)

        # Residual connection - ensure shapes match
        output = output + residual

        return output


# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape to 2D for efficient processing
            batch_size, seq_len, feat_dim = x.shape
            x = x.reshape(-1, feat_dim)

        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.reshape(original_shape)

        return x + residual


# Improved DeepTabAttention for tabular data
class DeepTabAttention(nn.Module):
    def __init__(self, input_dim, num_classes=1, num_transformer_blocks=4,
                 attention_dim=256, num_heads=8, ff_hidden_dim=512,
                 dropout=0.25):
        super(DeepTabAttention, self).__init__()

        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Create feature embeddings
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU()
        )

        # Create transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(attention_dim, attention_dim, num_heads, dropout),
                FeedForward(attention_dim, ff_hidden_dim, dropout)
            ]) for _ in range(num_transformer_blocks)
        ])

        # Final prediction head with multiple layers
        self.prediction_head = nn.Sequential(
            nn.Linear(attention_dim, ff_hidden_dim),
            nn.LayerNorm(ff_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.LayerNorm(ff_hidden_dim // 2),
            nn.Dropout(dropout / 2),
            nn.GELU(),
            nn.Linear(ff_hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Apply Xavier initialization for better convergence
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # Initial normalization
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Create embeddings
        x = self.feature_embedding(x)

        # For attention mechanism, reshape to add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, attention_dim)

        # Process through transformer blocks
        for attention, feed_forward in self.transformer_blocks:
            x = attention(x)
            x = feed_forward(x)

        # Pool along the sequence dimension (just squeeze since it's 1)
        x = x.squeeze(1)

        # Final prediction
        logits = self.prediction_head(x)

        return logits


# Ultra Deep Tabular with improved residual blocks
class UltraDeepTabular(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rates=[0.3, 0.3, 0.2, 0.2]):
        super(UltraDeepTabular, self).__init__()

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)

        # Create network layers
        self.layers = nn.ModuleList()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rates[0])
        )

        # Residual blocks
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(self._make_residual_block(
                hidden_sizes[i],
                hidden_sizes[i + 1],
                dropout_rates[i + 1]
            ))

        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], 1)

        # Initialize weights properly
        self._init_weights()

    def _make_residual_block(self, in_size, out_size, dropout_rate):
        """Create a residual block with skip connection"""
        layers = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_size, out_size),
            nn.BatchNorm1d(out_size)
        )

        # Add a skip connection if dimensions match, otherwise add a projection
        if in_size == out_size:
            skip = nn.Identity()
        else:
            skip = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size)
            )

        return nn.ModuleDict({
            'layers': layers,
            'skip': skip,
            'activation': nn.GELU(),
            'dropout': nn.Dropout(dropout_rate / 2)
        })

    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)

        # Input projection
        x = self.input_projection(x)

        # Residual blocks
        for block in self.layers:
            residual = block['skip'](x)
            x = block['layers'](x)
            x = x + residual  # Skip connection
            x = block['activation'](x)
            x = block['dropout'](x)

        # Final output
        return self.output(x)


# Fixed Mixture of Experts Network with proper expert routing
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, num_experts=4, expert_size=128, hidden_sizes=[64, 32]):
        super(MixtureOfExperts, self).__init__()

        self.input_size = input_size
        self.num_experts = num_experts
        self.expert_size = expert_size

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)

        # Gating network determines which expert to use
        self.gating_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )

        # Create multiple expert networks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = self._create_expert(input_size, expert_size, hidden_sizes)
            self.experts.append(expert)

        # Output layer combines expert outputs
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def _create_expert(self, input_size, expert_size, hidden_sizes):
        """Create a single expert network with improved architecture"""
        layers = []

        # Input layer with batch normalization
        layers.append(nn.Linear(input_size, expert_size))
        layers.append(nn.BatchNorm1d(expert_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))

        # Hidden layers
        prev_size = expert_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = size

        return nn.Sequential(*layers)

    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)

        # Get gating weights for each expert
        gating_weights = self.gating_network(x)

        # Process inputs through each expert separately
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # Shape: [batch_size, hidden_sizes[-1]]
            expert_outputs.append(expert_output)

        # Stack expert outputs - shape: [batch_size, num_experts, hidden_size]
        stacked_experts = torch.stack(expert_outputs, dim=1)

        # Apply gating weights - shape: [batch_size, num_experts, 1]
        gating_weights = gating_weights.unsqueeze(-1)

        # Weighted sum of expert outputs - shape: [batch_size, hidden_size]
        combined_output = torch.sum(stacked_experts * gating_weights, dim=1)

        # Final output layer
        return self.output_layer(combined_output)

    def get_expert_weights(self, x):
        """Return the gating weights for each expert for interpretability"""
        x = self.input_norm(x)
        return self.gating_network(x)


# Train XGBoost model - new addition
def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost...")
    start_time = time.time()

    # Calculate class weights
    class_counts = np.bincount(y_train)
    weight_scale = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0

    # Set scale_pos_weight for imbalanced classes
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': weight_scale,
        'tree_method': 'hist',  # for faster training
        'random_state': 42
    }

    # Create DMatrix for faster processing
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train with early stopping
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=50
    )

    training_time = time.time() - start_time
    print(f"XGBoost training completed in {training_time:.2f} seconds")

    # Make predictions
    train_preds_proba = model.predict(dtrain)
    train_preds = (train_preds_proba > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, train_preds)

    test_preds_proba = model.predict(dtest)
    test_preds = (test_preds_proba > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_preds_proba)

    # Get feature importances
    importance_scores = model.get_score(importance_type='gain')

    # Convert to a format compatible with sklearn models
    feature_importances = np.zeros(X_train.shape[1])
    for key, value in importance_scores.items():
        try:
            idx = int(key.replace('f', ''))
            feature_importances[idx] = value
        except (ValueError, IndexError):
            # Skip features that don't follow the expected naming pattern
            continue

    # Normalize importances
    if feature_importances.sum() > 0:
        feature_importances = feature_importances / feature_importances.sum()

    # Store the importances in the model object
    model.feature_importances_ = feature_importances

    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'test_predictions_proba': test_preds_proba,
        'training_time': training_time,
        'roc_auc': roc_auc,
        'evals_result': evals_result
    }


# Train SVM model - new addition
def train_svm(X_train, y_train, X_test, y_test):
    print("Training SVM with RBF kernel...")
    start_time = time.time()

    # Apply additional standardization for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    class_weight = {0: 1.0, 1: class_counts[0] / class_counts[1]} if class_counts[1] > 0 else {0: 1.0, 1: 1.0}

    # Use RBF kernel which often works well for tabular data
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight=class_weight,
        random_state=42,
        verbose=False
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time
    print(f"SVM training completed in {training_time:.2f} seconds")

    # Make predictions
    train_preds = model.predict(X_train_scaled)
    train_preds_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_accuracy = accuracy_score(y_train, train_preds)

    test_preds = model.predict(X_test_scaled)
    test_preds_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_accuracy = accuracy_score(y_test, test_preds)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_preds_proba)

    return {
        'model': model,
        'scaler': scaler,  # Need to save the scaler for future predictions
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'test_predictions_proba': test_preds_proba,
        'training_time': training_time,
        'roc_auc': roc_auc
    }


# Train Random Forest
def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    start_time = time.time()

    # Initialize and train the model with optimal hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,  # Let trees grow fully
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Better for generalization
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Random Forest training completed in {training_time:.2f} seconds")

    # Make predictions
    train_preds = model.predict(X_train)
    train_preds_proba = model.predict_proba(X_train)[:, 1]
    train_accuracy = accuracy_score(y_train, train_preds)

    test_preds = model.predict(X_test)
    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, test_preds)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_preds_proba)

    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'test_predictions_proba': test_preds_proba,
        'training_time': training_time,
        'roc_auc': roc_auc
    }


# Train DeepTabAttention with improved parameters
def train_deep_tab_attention(X_train, y_train, X_test, y_test, epochs=800, batch_size=256, patience=50):
    print("Training DeepTabAttention with extended epochs...")

    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model with improved architecture
    input_dim = X_train.shape[1]
    model = DeepTabAttention(
        input_dim=input_dim,
        num_classes=1,
        num_transformer_blocks=4,  # Increased from 3
        attention_dim=256,  # Increased from 192
        num_heads=8,  # Increased from 6
        ff_hidden_dim=512,  # Increased from 384
        dropout=0.25
    ).to(device)

    # Use focal loss for imbalanced classification
    def focal_loss(predictions, targets, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # probabilities
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * bce_loss).mean()

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # OneCycleLR scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Spend 30% of time warming up
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

                # Forward pass
                logits = model(inputs)
                loss = focal_loss(logits, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # Training metrics
                epoch_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # Calculate training metrics
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                    logits = model(inputs)
                    loss = focal_loss(logits, labels)
                    val_loss += loss.item()

                    # Validation metrics
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            # Calculate validation metrics
            avg_val_loss = val_loss / len(test_loader)
            val_accuracy = val_correct / val_total

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            # Early stopping - monitor both loss and accuracy
            improved = False

            if val_accuracy > best_val_acc + 0.001:  # 0.1% improvement threshold
                best_val_acc = val_accuracy
                improved = True

            if avg_val_loss < best_val_loss * 0.99:  # 1% improvement threshold
                best_val_loss = avg_val_loss
                improved = True

            if improved:
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    except Exception as e:
        print(f"Error during DeepTabAttention training: {e}")
        print("Attempting to continue with best model so far...")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    print(f"DeepTabAttention training completed in {training_time:.2f} seconds")

    # Calculate final predictions for evaluation
    model.eval()

    # Train predictions
    train_preds = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            train_preds.extend(probs.cpu().numpy().flatten())

    train_preds = np.array(train_preds)
    train_preds_binary = (train_preds > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, train_preds_binary)

    # Test predictions
    test_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy().flatten())

    test_preds = np.array(test_preds)
    test_preds_binary = (test_preds > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds_binary)

    roc_auc = roc_auc_score(y_test, test_preds)

    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds_binary,
        'test_predictions': test_preds_binary,
        'train_predictions_proba': train_preds,
        'test_predictions_proba': test_preds,
        'training_time': training_time,
        'history': history,
        'roc_auc': roc_auc
    }


# Improved UltraDeepTabular training
def train_ultra_deep_tabular(X_train, y_train, X_test, y_test, epochs=1000, batch_size=256, patience=50):
    print("Training UltraDeepTabular with extended epochs and advanced techniques...")

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TabularDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    # Initialize model
    model = UltraDeepTabular(X_train.shape[1]).to(device)

    # Loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[1] / class_weights[0]]).to(device))

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # One-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Training loop with early stopping and SWA
    start_time = time.time()

    # Variables for early stopping and tracking
    best_val_loss = float('inf')
    best_val_acc = 0
    early_stop_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Stochastic Weight Averaging
    swa_start = int(epochs * 0.75)  # Start SWA at 75% of training
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=1e-4)
    swa_active = False

    print(f"Planning to train for up to {epochs} epochs with patience {patience}")
    print(f"Will start Stochastic Weight Averaging after epoch {swa_start}")

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Update learning rate
                if not swa_active:
                    scheduler.step()

                # Calculate accuracy
                running_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # Calculate average training metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate validation accuracy
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_acc = val_correct / val_total

            # Store history
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)

            # Check if we should start SWA
            if epoch >= swa_start and not swa_active:
                print(f"Epoch {epoch + 1}: Starting Stochastic Weight Averaging")
                swa_active = True

            # Update SWA if active
            if swa_active:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

            # Early stopping - with thresholds for improvement
            improved = False

            if epoch_val_acc > best_val_acc + 0.001:  # 0.1% improvement threshold
                best_val_acc = epoch_val_acc
                improved = True
                print(f"Improved validation accuracy: {best_val_acc:.4f}")

            if epoch_val_loss < best_val_loss * 0.99:  # 1% improvement threshold
                best_val_loss = epoch_val_loss
                improved = True
                print(f"Improved validation loss: {best_val_loss:.4f}")

            if not swa_active and improved:
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            elif not swa_active:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    except Exception as e:
        print(f"Error during training: {e}")
        print("Attempting to continue with best model so far...")

    # Finalize SWA if active
    if swa_active:
        print("Finalizing SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        final_model = swa_model
    else:
        # Load best model if we're not using SWA
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        final_model = model

    training_time = time.time() - start_time
    print(f"UltraDeepTabular training completed in {training_time:.2f} seconds")

    # Evaluate the model
    final_model.eval()

    # Get predictions for training data
    train_preds = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = final_model(inputs)
            probs = torch.sigmoid(outputs)
            train_preds.extend(probs.cpu().numpy().flatten())

    train_preds = np.array(train_preds)
    train_preds_binary = (train_preds > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, train_preds_binary)

    # Get predictions for test data
    test_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = final_model(inputs)
            probs = torch.sigmoid(outputs)
            test_preds.extend(probs.cpu().numpy().flatten())

    test_preds = np.array(test_preds)
    test_preds_binary = (test_preds > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds_binary)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_preds)

    return {
        'model': final_model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds_binary,
        'test_predictions': test_preds_binary,
        'train_predictions_proba': train_preds,
        'test_predictions_proba': test_preds,
        'training_time': training_time,
        'history': history,
        'roc_auc': roc_auc
    }


# Improved Mixture of Experts training
def train_mixture_of_experts(X_train, y_train, X_test, y_test, epochs=600, batch_size=256, patience=40):
    print("Training Mixture of Experts Model...")

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TabularDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model with more experts
    model = MixtureOfExperts(X_train.shape[1], num_experts=5).to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler with warm-up and cosine annealing
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    start_time = time.time()

    # Variables for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    early_stop_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # Calculate accuracy
                running_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # Calculate average training metrics
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate validation accuracy
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_acc = val_correct / val_total

            # Store history
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

            # Early stopping - with improvement thresholds
            improved = False

            if epoch_val_acc > best_val_acc + 0.001:  # 0.1% improvement threshold
                best_val_acc = epoch_val_acc
                improved = True

            if epoch_val_loss < best_val_loss * 0.99:  # 1% improvement threshold
                best_val_loss = epoch_val_loss
                improved = True

            if improved:
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    except Exception as e:
        print(f"Error during training: {e}")
        print("Attempting to continue with best model so far...")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time
    print(f"Mixture of Experts training completed in {training_time:.2f} seconds")

    # Evaluate the model
    model.eval()

    # Get final predictions on full datasets
    train_preds = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            train_preds.extend(probs.cpu().numpy().flatten())

    train_preds = np.array(train_preds)
    train_preds_binary = (train_preds > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, train_preds_binary)

    # Get predictions for test data and expert weights
    test_preds = []
    expert_weights_list = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            test_preds.extend(probs.cpu().numpy().flatten())

            # Get expert weights for interpretability
            expert_weights = model.get_expert_weights(inputs)
            expert_weights_list.append(expert_weights.cpu().numpy())

    test_preds = np.array(test_preds)
    test_preds_binary = (test_preds > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds_binary)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, test_preds)

    # Calculate average expert usage
    all_expert_weights = np.vstack(expert_weights_list)
    avg_expert_weights = all_expert_weights.mean(axis=0)

    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_preds_binary,
        'test_predictions': test_preds_binary,
        'train_predictions_proba': train_preds,
        'test_predictions_proba': test_preds,
        'training_time': training_time,
        'history': history,
        'expert_weights': all_expert_weights,
        'avg_expert_weights': avg_expert_weights,
        'roc_auc': roc_auc
    }


# Create a voting ensemble from the trained models
def create_ensemble(models_dict, X_test, y_test):
    print("Creating ensemble model...")

    # Extract predictions from each model
    all_preds_proba = {}
    for model_name, model_result in models_dict.items():
        if 'test_predictions_proba' in model_result:
            all_preds_proba[model_name] = model_result['test_predictions_proba']

    if not all_preds_proba:
        print("No probability predictions available for ensemble")
        return None

    # Weighted averaging based on ROC AUC scores
    ensemble_weights = {}
    for model_name, model_result in models_dict.items():
        if 'roc_auc' in model_result:
            # Weight by AUC squared to emphasize better models
            ensemble_weights[model_name] = model_result['roc_auc'] ** 2

    # Normalize weights
    total_weight = sum(ensemble_weights.values())
    for model_name in ensemble_weights:
        ensemble_weights[model_name] /= total_weight

    print("Ensemble weights:")
    for model_name, weight in ensemble_weights.items():
        print(f"  {model_name}: {weight:.4f}")

    # Compute weighted average predictions
    ensemble_preds_proba = np.zeros_like(next(iter(all_preds_proba.values())))
    for model_name, preds_proba in all_preds_proba.items():
        if model_name in ensemble_weights:
            ensemble_preds_proba += preds_proba * ensemble_weights[model_name]

    # Convert to binary predictions
    ensemble_preds = (ensemble_preds_proba > 0.5).astype(int)

    # Evaluate ensemble
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    ensemble_roc_auc = roc_auc_score(y_test, ensemble_preds_proba)

    print(f"Ensemble Results:")
    print(f"  Accuracy: {ensemble_accuracy:.4f}")
    print(f"  ROC AUC: {ensemble_roc_auc:.4f}")

    ensemble_result = {
        'test_accuracy': ensemble_accuracy,
        'test_predictions': ensemble_preds,
        'test_predictions_proba': ensemble_preds_proba,
        'roc_auc': ensemble_roc_auc,
        'weights': ensemble_weights
    }

    return ensemble_result


# Function to evaluate and report model performance
def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # Calculate ROC AUC if probability predictions are available
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    if roc_auc is not None:
        print(f"ROC AUC Score: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'roc_auc': roc_auc
    }


# Function to visualize model comparison
def visualize_model_comparison(results):
    models = list(results.keys())
    train_acc = [results[model].get('train_accuracy', 0) for model in models]
    test_acc = [results[model]['test_accuracy'] for model in models]
    training_times = [results[model].get('training_time', 0) for model in models]
    roc_auc_scores = [results[model].get('roc_auc', 0) for model in models]

    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))

    # Plot accuracy comparison
    x = np.arange(len(models))
    width = 0.35

    bar1 = axs[0].bar(x - width / 2, train_acc, width, label='Train Accuracy', color='skyblue')
    bar2 = axs[0].bar(x + width / 2, test_acc, width, label='Test Accuracy', color='lightcoral')
    axs[0].set_ylabel('Accuracy', fontsize=12)
    axs[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    axs[0].legend(fontsize=10)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Add accuracy values as text
    for bar in bar1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.4f}', ha='center', fontsize=9)

    for bar in bar2:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.4f}', ha='center', fontsize=9)

    # Plot ROC AUC scores
    if any(roc_auc_scores):
        bar3 = axs[1].bar(x, roc_auc_scores, width, color='lightgreen')
        axs[1].set_ylabel('ROC AUC Score', fontsize=12)
        axs[1].set_title('Model ROC AUC Comparison', fontsize=14, fontweight='bold')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Add AUC values as text
        for bar in bar3:
            height = bar.get_height()
            if height > 0:
                axs[1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.4f}', ha='center', fontsize=9)

    # Plot training time comparison
    bar4 = axs[2].bar(x, training_times, width, color='lightsalmon')
    axs[2].set_ylabel('Training Time (seconds)', fontsize=12)
    axs[2].set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Add training time values as text
    for bar in bar4:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{height:.2f}s', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison chart to 'model_comparison.png'")

    # Plot correlation between test accuracy and training time
    plt.figure(figsize=(10, 6))
    plt.scatter(training_times, test_acc, s=100, alpha=0.7)

    # Add model names as labels
    for i, model in enumerate(models):
        plt.annotate(model, (training_times[i], test_acc[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Test Accuracy vs Training Time', fontsize=14, fontweight='bold')
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('accuracy_vs_time.png', dpi=300, bbox_inches='tight')
    print("Saved correlation chart to 'accuracy_vs_time.png'")


# Enhanced main function
def main():
    # Set the file path
    file_path = 'Person_number.csv'

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load and preprocess data
    df = load_and_preprocess_data(file_path)

    # Prepare data for modeling with feature engineering and SMOTE
    X_train, X_test, y_train, y_test, numerical_features, categorical_features, preprocessor = prepare_data(df)

    # Dictionary to store model results
    results = {}

    # Train Random Forest - the best performer
    try:
        rf_results = train_random_forest(X_train, y_train, X_test, y_test)
        rf_eval = evaluate_model(y_test, rf_results['test_predictions'],
                                 rf_results['test_predictions_proba'], "Random Forest")
        results['Random Forest'] = {**rf_results, **rf_eval}
    except Exception as e:
        print(f"Error training Random Forest: {e}")

    # Train XGBoost - new addition
    try:
        xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
        xgb_eval = evaluate_model(y_test, xgb_results['test_predictions'],
                                  xgb_results['test_predictions_proba'], "XGBoost")
        results['XGBoost'] = {**xgb_results, **xgb_eval}
    except Exception as e:
        print(f"Error training XGBoost: {e}")
        print("Make sure XGBoost is installed with: pip install xgboost")

    # Train SVM - new addition
    try:
        svm_results = train_svm(X_train, y_train, X_test, y_test)
        svm_eval = evaluate_model(y_test, svm_results['test_predictions'],
                                  svm_results['test_predictions_proba'], "SVM")
        results['SVM'] = {**svm_results, **svm_eval}
    except Exception as e:
        print(f"Error training SVM: {e}")

    # Train DeepTabAttention
    try:
        dta_results = train_deep_tab_attention(X_train, y_train, X_test, y_test)
        dta_eval = evaluate_model(y_test, dta_results['test_predictions'],
                                  dta_results['test_predictions_proba'], "DeepTabAttention")
        results['DeepTabAttention'] = {**dta_results, **dta_eval}
    except Exception as e:
        print(f"Error training DeepTabAttention: {e}")

    # Train UltraDeepTabular
    try:
        udt_results = train_ultra_deep_tabular(X_train, y_train, X_test, y_test)
        udt_eval = evaluate_model(y_test, udt_results['test_predictions'],
                                  udt_results['test_predictions_proba'], "UltraDeepTabular")
        results['UltraDeepTabular'] = {**udt_results, **udt_eval}
    except Exception as e:
        print(f"Error training UltraDeepTabular: {e}")

    # Train Mixture of Experts
    try:
        moe_results = train_mixture_of_experts(X_train, y_train, X_test, y_test)
        moe_eval = evaluate_model(y_test, moe_results['test_predictions'],
                                  moe_results['test_predictions_proba'], "Mixture of Experts")
        results['Mixture of Experts'] = {**moe_results, **moe_eval}
    except Exception as e:
        print(f"Error training Mixture of Experts: {e}")

    # Create ensemble model
    try:
        ensemble_result = create_ensemble(results, X_test, y_test)
        if ensemble_result:
            ensemble_eval = evaluate_model(y_test, ensemble_result['test_predictions'],
                                           ensemble_result['test_predictions_proba'], "Ensemble")
            results['Ensemble'] = {**ensemble_result, **ensemble_eval}
    except Exception as e:
        print(f"Error creating ensemble: {e}")

    # Visualize model comparison
    if results:
        visualize_model_comparison(results)

        # Print summary
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        print("{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            "Model", "Train Accuracy", "Test Accuracy", "Training Time", "ROC AUC"))
        print("-" * 80)

        for model, result in results.items():
            train_acc = result.get('train_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            training_time = result.get('training_time', 0)
            roc_auc = result.get('roc_auc', 0)

            print("{:<20} {:<15.4f} {:<15.4f} {:<15.2f}s {:<15.4f}".format(
                model, train_acc, test_acc, training_time, roc_auc
            ))

        # Identify the best model based on test accuracy
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print("\nBest performing model (by accuracy):", best_model[0])
        print(f"Test Accuracy: {best_model[1]['test_accuracy']:.4f}")

        # Also identify best model by ROC AUC if available
        if any('roc_auc' in result for result in results.values()):
            best_model_auc = max(
                [(k, v) for k, v in results.items() if 'roc_auc' in v],
                key=lambda x: x[1]['roc_auc']
            )
            print(f"\nBest performing model (by AUC): {best_model_auc[0]}")
            print(f"ROC AUC: {best_model_auc[1]['roc_auc']:.4f}")

        # Feature importance analysis for the best model (if available)
        if best_model[0] != 'Ensemble' and hasattr(best_model[1]['model'], 'feature_importances_'):
            print("\nTop 20 Most Important Features:")

            importances = best_model[1]['model'].feature_importances_
            if len(importances) == len(X_train.columns):
                feature_names = X_train.columns
            else:
                # For models that may have different feature representations
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Sort feature importances
            indices = np.argsort(importances)[::-1]

            # Print top 20 features
            for i in range(min(20, len(feature_names))):
                if i < len(indices):
                    idx = indices[i]
                    if idx < len(feature_names):
                        print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

            # Visualize top features
            plt.figure(figsize=(12, 8))
            plt.title(f"Top 20 Feature Importances - {best_model[0]}")

            # Plot at most 20 features
            num_features = min(20, len(indices), len(feature_names))
            display_indices = indices[:num_features]

            plt.bar(range(num_features),
                    [importances[idx] for idx in display_indices])

            # Handle long feature names with rotation
            feature_labels = [feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                              for idx in display_indices]

            plt.xticks(range(num_features), feature_labels, rotation=90)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("Saved feature importance chart to 'feature_importance.png'")
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main()