import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
import os


class TourDataset(Dataset):
    """Dataset for tour prediction"""

    def __init__(self, categorical_features, numerical_features, toured_labels, lead_ids=None):
        self.categorical_features = categorical_features.long() if categorical_features is not None else None
        self.numerical_features = numerical_features.float() if numerical_features is not None else None
        self.toured_labels = toured_labels.float() if toured_labels is not None else None
        self.lead_ids = lead_ids

    def __len__(self):
        return len(self.toured_labels)

    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([],
                                                                                                      dtype=torch.long),
            self.numerical_features[idx] if self.numerical_features is not None else torch.tensor([],
                                                                                                  dtype=torch.float),
            self.toured_labels[idx]
        ]

        if self.lead_ids is not None:
            items.append(self.lead_ids[idx].clone().detach())

        return tuple(items)


def prepare_tour_data_simplified(
        data_path,
        target_col='TOTAL_APPOINTMENT_COMPLETED',
        test_size=0.2,
        random_state=42,
        max_categories=100,
        enhance_features=False  # Set default to False for simplicity
):
    """
    Simplified data preparation pipeline for tour prediction
    """
    print("\nLoading and preprocessing data...")

    # 1. Load Data
    data = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded data with {len(data):,} rows and {len(data.columns):,} columns")

    # 2. Create Target Variable
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    y_toured = (data[target_col] > 0).astype(int)
    print(f"Target (Toured): {y_toured.mean() * 100:.2f}% positive ({y_toured.sum():,} of {len(y_toured):,})")

    # 3. Drop irrelevant/leaky columns
    # Simplified list of columns to drop - customize based on your dataset
    cols_to_drop = [
        # Identifiers and personal info
        "HASH", "RECD_LUID", "RECD_P_ID", "RECD_A_ID", "CLIENT_PERSON_ID", "CLIENT_ID", "RN",
        "FNAM_FNAM", "MNAM_MNAM", "PFXT_PFXT", "SNAM_SNAM", "SFXT_SFXT", "FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE",

        # Address info
        "ADDRESS_LINE_1", "ADDRESS_LINE_2", "CITY", "STRT_NAME_I1", "STRT_POST_I1", "STRT_PRED_I1", "STRT_SUFX_I1",

        # Dates and timestamps (can create high cardinality)
        "EXTRACT_DATE", "NCOA_MOVE_UPDATE_DATE",

        # Obvious target leakage
        "QUALIFIED", "TOTAL_WALK_IN",

        # Future events that would cause leakage
        "TOTAL_APPLIED", "TOTAL_RENTED"
    ]

    # Extract lead IDs before dropping
    id_col_name = 'CLIENT_PERSON_ID'  # Adjust based on your data
    if id_col_name in data.columns:
        lead_ids = data[id_col_name].copy()
    else:
        lead_ids = pd.Series(np.arange(len(data)), name='generated_id')
        id_col_name = 'generated_id'

    # Drop specified columns
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

    # Drop columns with >80% missing values (more aggressive than the original 95%)
    missing_pct = data.isnull().mean()
    cols_to_drop_missing = missing_pct[missing_pct > 0.8].index
    data = data.drop(cols_to_drop_missing, axis=1)
    print(f"Dropped {len(cols_to_drop_missing)} columns with >80% missing values")

    # 4. Feature Engineering (minimal compared to original)
    if enhance_features:
        # Identify date columns for basic time features
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            try:
                # Try to convert to datetime
                dt_col = pd.to_datetime(data[col], errors='coerce')
                if not dt_col.isnull().all():
                    # Create day of week feature
                    data[f'{col}_dow'] = dt_col.dt.dayofweek
                    # Create weekend flag
                    data[f'{col}_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
                    print(f"Created time features from '{col}'")
            except:
                pass

    # 5. Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in data.columns:
        n_unique = data[col].nunique()

        # Handle potential boolean values
        if data[col].dtype == 'object':
            # Try to convert to numeric
            try:
                num_col = pd.to_numeric(data[col], errors='coerce')
                if num_col.notna().mean() > 0.9:  # If most values converted successfully
                    data[col] = num_col
            except:
                pass

        # Determine column type
        if (data[col].dtype == 'object' or
                data[col].dtype == 'category' or
                (pd.api.types.is_integer_dtype(data[col]) and n_unique <= 20)):
            # Categorical if it's object, category, or integer with ≤20 unique values
            if n_unique <= max_categories:
                categorical_cols.append(col)
            else:
                print(f"Dropping high-cardinality column '{col}' with {n_unique} values")
        elif pd.api.types.is_numeric_dtype(data[col]):
            numerical_cols.append(col)

    print(f"Identified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")

    # 6. Handle missing values (simplified)
    # For categorical: fill with 'missing'
    for col in categorical_cols:
        data[col] = data[col].fillna('missing')

    # For numerical: fill with median
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())

    # 7. Train/Test Split
    X = data[categorical_cols + numerical_cols]

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X, y_toured, lead_ids, test_size=test_size, random_state=random_state,
        stratify=y_toured if y_toured.nunique() > 1 else None
    )

    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"Train target: {y_train.mean() * 100:.2f}% positive, Test target: {y_test.mean() * 100:.2f}% positive")

    # 8. Preprocessing
    # Encode categorical features
    categorical_encoders = {}
    categorical_dims = []
    X_train_cat_encoded = pd.DataFrame(index=X_train.index)
    X_test_cat_encoded = pd.DataFrame(index=X_test.index)

    for col in categorical_cols:
        encoder = LabelEncoder()
        # Fit on both train and test to avoid unseen categories
        combined_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        encoder.fit(combined_values)

        # Transform
        X_train_cat_encoded[col] = encoder.transform(X_train[col].astype(str))
        X_test_cat_encoded[col] = encoder.transform(X_test[col].astype(str))

        # Store encoder and dimensions
        categorical_encoders[col] = encoder
        categorical_dims.append(len(encoder.classes_))

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = X_train[numerical_cols]
    X_test_num = X_test[numerical_cols]

    if len(numerical_cols) > 0:
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_test_num_scaled = scaler.transform(X_test_num)
    else:
        X_train_num_scaled = np.zeros((len(X_train), 0))
        X_test_num_scaled = np.zeros((len(X_test), 0))

    numerical_dim = X_train_num_scaled.shape[1]

    # 9. Create PyTorch tensors and datasets
    # Convert to tensors
    X_train_cat_tensor = torch.tensor(X_train_cat_encoded.values, dtype=torch.long) if len(
        categorical_cols) > 0 else torch.zeros((len(X_train), 0), dtype=torch.long)
    X_test_cat_tensor = torch.tensor(X_test_cat_encoded.values, dtype=torch.long) if len(
        categorical_cols) > 0 else torch.zeros((len(X_test), 0), dtype=torch.long)

    X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
    X_test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_ids_tensor = torch.tensor(train_ids.values) if hasattr(train_ids, 'values') else torch.tensor(train_ids)
    test_ids_tensor = torch.tensor(test_ids.values) if hasattr(test_ids, 'values') else torch.tensor(test_ids)

    # Create datasets
    train_dataset = TourDataset(X_train_cat_tensor, X_train_num_tensor, y_train_tensor, train_ids_tensor)
    test_dataset = TourDataset(X_test_cat_tensor, X_test_num_tensor, y_test_tensor, test_ids_tensor)

    print("Data preparation complete!")

    return (
        train_dataset,
        test_dataset,
        categorical_dims,
        numerical_dim,
        categorical_cols + numerical_cols
    )