import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import category_encoders as ce
from tqdm import tqdm
import pickle
import os


def preprocess_data(data_path, target_cols=['TOTAL_WALK_IN', 'QUALIFIED', 'TOTAL_RENTED'],
                    test_size=0.2, random_state=42, max_categories=100,
                    save_preprocessors=True, preprocessors_path='./preprocessors'):
    """
    Preprocess the lead data for the neural network model

    Args:
        data_path: Path to the CSV file containing lead data
        target_cols: List of column names for tour, apply, and rent targets
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        max_categories: Maximum number of categories to consider for categorical variables
        save_preprocessors: Whether to save preprocessors for later use
        preprocessors_path: Path to save preprocessors

    Returns:
        train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names
    """
    print("Loading data...")
    data = pd.read_csv(data_path)

    # Create binary target variables
    print("Creating target variables...")
    y_tour = (data[target_cols[0]] > 0).astype(int)
    y_apply = (data[target_cols[1]] > 0).astype(int)
    y_rent = (data[target_cols[2]] > 0).astype(int)

    # Drop target columns and any ID columns that shouldn't be features
    print("Preparing feature set...")
    X = data.drop(target_cols, axis=1)

    # Remove any columns with >95% missing values
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.95].index
    X = X.drop(cols_to_drop, axis=1)
    print(f"Dropped {len(cols_to_drop)} columns with >95% missing values")

    # Separate ID column for later reference
    lead_ids = X['CLIENT_PERSON_ID'] if 'CLIENT_PERSON_ID' in X.columns else np.arange(len(X))

    # Remove ID columns and other non-feature columns
    id_columns = [col for col in X.columns if 'ID' in col.upper() and X[col].nunique() > 1000]
    X = X.drop(id_columns + ['CLIENT_PERSON_ID'] if 'CLIENT_PERSON_ID' in X.columns else id_columns, axis=1)

    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []

    for col in X.columns:
        if X[col].dtype == 'object' or (X[col].dtype in ['int64', 'float64'] and X[col].nunique() < 20):
            if X[col].nunique() <= max_categories:  # Filter out high cardinality
                categorical_cols.append(col)
        elif X[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)

    print(f"Identified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")

    # Handle missing values
    # For numerical: impute with median
    numerical_data = X[numerical_cols].copy()
    numerical_data = numerical_data.fillna(numerical_data.median())

    # For categorical: impute with mode
    categorical_data = X[categorical_cols].copy()
    for col in categorical_cols:
        categorical_data[col] = categorical_data[col].fillna(categorical_data[col].mode()[0])

    # Split the data
    print("Splitting into train/test sets...")
    X_train_cat, X_test_cat, X_train_num, X_test_num, y_train_tour, y_test_tour, y_train_apply, y_test_apply, y_train_rent, y_test_rent, train_ids, test_ids = train_test_split(
        categorical_data, numerical_data, y_tour, y_apply, y_rent, lead_ids,
        test_size=test_size, random_state=random_state, stratify=y_rent  # Stratify on the rarest outcome
    )

    # Process categorical features - use target encoding which works well for high cardinality
    print("Processing categorical features...")
    categorical_encoders = {}
    X_train_cat_encoded = pd.DataFrame()
    X_test_cat_encoded = pd.DataFrame()

    # For high cardinality categorical variables, use target encoding
    target_encoder = ce.TargetEncoder(cols=categorical_cols)
    X_train_cat_encoded = target_encoder.fit_transform(X_train_cat, y_train_rent)
    X_test_cat_encoded = target_encoder.transform(X_test_cat)

    # Process numerical features
    print("Processing numerical features...")
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # Convert to tensors
    X_train_cat_tensor = torch.tensor(X_train_cat_encoded.values, dtype=torch.float32)
    X_test_cat_tensor = torch.tensor(X_test_cat_encoded.values, dtype=torch.float32)
    X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
    X_test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32)

    y_train_tour_tensor = torch.tensor(y_train_tour.values, dtype=torch.float32).view(-1, 1)
    y_test_tour_tensor = torch.tensor(y_test_tour.values, dtype=torch.float32).view(-1, 1)
    y_train_apply_tensor = torch.tensor(y_train_apply.values, dtype=torch.float32).view(-1, 1)
    y_test_apply_tensor = torch.tensor(y_test_apply.values, dtype=torch.float32).view(-1, 1)
    y_train_rent_tensor = torch.tensor(y_train_rent.values, dtype=torch.float32).view(-1, 1)
    y_test_rent_tensor = torch.tensor(y_test_rent.values, dtype=torch.float32).view(-1, 1)

    # Create datasets
    from training_pipeline import LeadDataset

    train_dataset = LeadDataset(
        X_train_cat_tensor, X_train_num_tensor,
        y_train_tour_tensor, y_train_apply_tensor, y_train_rent_tensor,
        torch.tensor(train_ids.values if isinstance(train_ids, pd.Series) else train_ids)
    )

    test_dataset = LeadDataset(
        X_test_cat_tensor, X_test_num_tensor,
        y_test_tour_tensor, y_test_apply_tensor, y_test_rent_tensor,
        torch.tensor(test_ids.values if isinstance(test_ids, pd.Series) else test_ids)
    )

    # Save preprocessors for future use
    if save_preprocessors:
        if not os.path.exists(preprocessors_path):
            os.makedirs(preprocessors_path)

        preprocessors = {
            'target_encoder': target_encoder,
            'scaler': scaler,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'feature_names': X_train_cat_encoded.columns.tolist() + X_train_num.columns.tolist()
        }

        with open(os.path.join(preprocessors_path, 'preprocessors.pkl'), 'wb') as f:
            pickle.dump(preprocessors, f)

    # Calculate dimensions for model initialization
    categorical_dims = [1] * X_train_cat_encoded.shape[1]  # For target encoding, each column is treated as 1 dimension
    numerical_dim = X_train_num_scaled.shape[1]
    feature_names = X_train_cat_encoded.columns.tolist() + X_train_num.columns.tolist()

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Testing set: {len(test_dataset)} samples")

    return train_dataset, test_dataset, categorical_dims, numerical_dim, feature_names


def prepare_external_examples(external_data, preprocessors_path='./preprocessors'):
    """
    Prepare external examples for fine-tuning using saved preprocessors

    Args:
        external_data: DataFrame containing the external examples
        preprocessors_path: Path to the saved preprocessors

    Returns:
        external_dataset: Dataset containing processed external examples
    """
    # Load the preprocessors
    with open(os.path.join(preprocessors_path, 'preprocessors.pkl'), 'rb') as f:
        preprocessors = pickle.load(f)

    target_encoder = preprocessors['target_encoder']
    scaler = preprocessors['scaler']
    categorical_cols = preprocessors['categorical_cols']
    numerical_cols = preprocessors['numerical_cols']

    # Extract features and targets
    X = external_data.copy()
    y_tour = (X['TOTAL_WALK_IN'] > 0).astype(int)
    y_apply = (X['QUALIFIED'] > 0).astype(int)
    y_rent = (X['TOTAL_RENTED'] > 0).astype(int)

    # Remove target columns
    X = X.drop(['TOTAL_WALK_IN', 'QUALIFIED', 'TOTAL_RENTED'], axis=1)

    # Handle missing columns
    for col in categorical_cols:
        if col not in X.columns:
            X[col] = 'missing'

    for col in numerical_cols:
        if col not in X.columns:
            X[col] = 0

    # Extract categorical and numerical data
    categorical_data = X[categorical_cols].copy()
    numerical_data = X[numerical_cols].copy()

    # Handle missing values
    for col in categorical_cols:
        categorical_data[col] = categorical_data[col].fillna('missing')

    for col in numerical_cols:
        numerical_data[col] = numerical_data[col].fillna(0)

    # Apply the transformations
    categorical_encoded = target_encoder.transform(categorical_data)
    numerical_scaled = scaler.transform(numerical_data)

    # Convert to tensors
    X_cat_tensor = torch.tensor(categorical_encoded.values, dtype=torch.float32)
    X_num_tensor = torch.tensor(numerical_scaled, dtype=torch.float32)
    y_tour_tensor = torch.tensor(y_tour.values, dtype=torch.float32).view(-1, 1)
    y_apply_tensor = torch.tensor(y_apply.values, dtype=torch.float32).view(-1, 1)
    y_rent_tensor = torch.tensor(y_rent.values, dtype=torch.float32).view(-1, 1)

    # Create dataset
    from training_pipeline import LeadDataset
    external_dataset = LeadDataset(
        X_cat_tensor, X_num_tensor,
        y_tour_tensor, y_apply_tensor, y_rent_tensor
    )

    return external_dataset


def visualize_group_differences(differences, stage, save_path=None):
    """
    Visualize the differences between selected and non-selected leads for a stage

    Args:
        differences: Output from analyze_group_differences function
        stage: Name of the stage (tour, apply, rent)
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract data for plotting
    features = [item[0] for item in differences]
    abs_diffs = [item[1]['abs_diff'] for item in differences]

    # Create a color map based on the direction of difference
    colors = ['green' if diff > 0 else 'red' for diff in abs_diffs]

    # Create the plot
    plt.figure(figsize=(12, 10))
    bars = plt.barh(features, abs_diffs, color=colors)

    # Add labels and title
    plt.xlabel('Absolute Difference (Selected - Non-Selected)')
    plt.title(f'Top Feature Differences for {stage.capitalize()} Stage')

    # Add a horizontal line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Add annotations for the differences
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width >= 0:
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{width:.2f}", ha='left', va='center')
        else:
            plt.text(width - 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{width:.2f}", ha='right', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()