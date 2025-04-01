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

# Move the import to the top to avoid potential issues
from training_pipeline import LeadDataset


def preprocess_data(data_path, dict_path=None, dict_map_path=None,
                    target_cols=['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED'],
                    test_size=0.2, random_state=42, max_categories=100,
                    save_preprocessors=True, preprocessors_path='./preprocessors'):
    """
    Preprocess the lead data for the neural network model

    Args:
        data_path: Path to the CSV file containing lead data
        target_cols: List of column names for toured, applied, and rented targets
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

    # Load data dictionary if provided
    data_dict = None
    if dict_path and os.path.exists(dict_path):
        print("Loading data dictionary...")
        data_dict = pd.read_csv(dict_path)
        print(f"Loaded dictionary with {len(data_dict)} entries")

    # Load dictionary mapping if provided
    dict_map = None
    if dict_map_path and os.path.exists(dict_map_path):
        print("Loading dictionary mapping...")
        dict_map = pd.read_csv(dict_map_path)
        print(f"Loaded dictionary mapping with {len(dict_map)} entries")

    # Create binary target variables
    print("Creating target variables...")
    # Check if target columns exist in the dataset
    for col in target_cols:
        if col not in data.columns:
            raise ValueError(
                f"Target column '{col}' not found in dataset. Available columns: {', '.join(data.columns[:10])}...")

    y_toured = (data[target_cols[0]] > 0).astype(int)  # TOTAL_APPOINTMENT_COMPLETED (TOURED)
    y_applied = (data[target_cols[1]] > 0).astype(int)  # TOTAL_APPLIED (APPLIED)
    y_rent = (data[target_cols[2]] > 0).astype(int)  # TOTAL_RENTED (RENTED)

    # Use data dictionary to find important columns if available
    important_cols = []
    if data_dict is not None:
        # Find columns related to leads, customers, properties, etc.
        lead_keywords = ['lead', 'customer', 'client', 'property', 'apartment', 'rent', 'tour', 'visit']
        for keyword in lead_keywords:
            matching_fields = data_dict[data_dict['LONG_DESCRIPTION'].str.contains(keyword, case=False, na=False)]
            important_cols.extend(matching_fields['FIELD_NAME'].tolist())

        print(f"Identified {len(important_cols)} potentially important columns from data dictionary")

    # Drop target columns and any ID columns that shouldn't be features
    print("Preparing feature set...")
    X = data.drop(target_cols, axis=1)

    # Drop specified columns (from previous analysis)
    cols_to_drop = [
        "HASH", "RECD_LUID", "RECD_P_ID", "RECD_A_ID", "CLIENT_PERSON_ID", "CLIENT_ID", "RN",
        "FNAM_FNAM", "MNAM_MNAM", "PFXT_PFXT", "SNAM_SNAM", "SFXT_SFXT", "FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE",
        "ADDRESS_LINE_1", "ADDRESS_LINE_2", "CITY", "STRT_NAME_I1", "STRT_POST_I1", "STRT_PRED_I1", "STRT_SUFX_I1",
        "QUALIFIED",
        "EXTRACT_DATE", "NCOA_MOVE_UPDATE_DATE", "NCOA_MOVE_UPDATE_METHOD_CODE",
        "EST_CURRENT_MORTGAGE_AMOUNT", "ENRICHMENTESTIMATED_CURRENT_MORTGAGE_AMT", "EST_MONTHLY_MORTGAGE_PAYMENT",
        "EST_CURRENT_LOAN-TO-VALUE_RATIO", "EST_AVAILABLE_EQUITY_LL", "ESTIMATED_AVAILABLE_EQUITY",
        "EQTY_LNDR_I1", "MORT_LNDR_I1", "REFI_LNDR_I1",
        "GROUP_ID", "GROUP_NAME", "LEAD_CREATED_AT", "CITY_PLAC_I1", "COUNTY_CODE_I1", "STATE_ABBR_I1", "STATE_I1",
        "RECD_ZIPC_I1",
        "SCDY_NUMB_I1", "SCDY_DESG_I1", "INVS_TYPE_I1", "PRCH_TYPE_I1",
        "TOTAL_WALK_IN"  # Explicitly drop TOTAL_WALK_IN as it’s meaningless
    ]
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])

    # Remove any columns with >95% missing values
    missing_pct = X.isnull().mean()
    cols_to_drop_missing = missing_pct[missing_pct > 0.95].index
    X = X.drop(cols_to_drop_missing, axis=1)
    print(f"Dropped {len(cols_to_drop_missing)} columns with >95% missing values")

    # --- Temporal filtering removed ---
    # The entire block using LEAD_CREATED_AT has been removed.

    # Separate ID column for later reference
    lead_ids = X['CLIENT_PERSON_ID'] if 'CLIENT_PERSON_ID' in X.columns else np.arange(len(X))

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
    X_train_cat, X_test_cat, X_train_num, X_test_num, y_train_toured, y_test_toured, y_train_applied, y_test_applied, y_train_rent, y_test_rent, train_ids, test_ids = train_test_split(
        categorical_data, numerical_data, y_toured, y_applied, y_rent, lead_ids,
        test_size=test_size, random_state=random_state, stratify=y_rent  # Stratify on the rarest outcome
    )

    # Optimize data loading for large dataset
    print("\nOptimizing data processing for large dataset...")
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Process categorical features - use target encoding which works well for high cardinality
    print("Processing categorical features...")
    categorical_encoders = {}
    X_train_cat_encoded = pd.DataFrame()
    X_test_cat_encoded = pd.DataFrame()

    # Process in smaller chunks for high cardinality categorical variables
    chunk_size = 50  # Process 50 columns at a time
    for i in range(0, len(categorical_cols), chunk_size):
        chunk_cols = categorical_cols[i:i + chunk_size]
        print(
            f"Processing categorical chunk {i // chunk_size + 1}/{(len(categorical_cols) - 1) // chunk_size + 1} ({len(chunk_cols)} columns)")

        # Use target encoding for high cardinality categorical features
        chunk_encoder = ce.TargetEncoder(cols=chunk_cols)
        chunk_train_encoded = chunk_encoder.fit_transform(X_train_cat[chunk_cols], y_train_rent)
        chunk_test_encoded = chunk_encoder.transform(X_test_cat[chunk_cols])

        # Concatenate with previous chunks
        X_train_cat_encoded = pd.concat([X_train_cat_encoded, chunk_train_encoded], axis=1)
        X_test_cat_encoded = pd.concat([X_test_cat_encoded, chunk_test_encoded], axis=1)

        # Save encoder for this chunk
        categorical_encoders[f"chunk_{i // chunk_size}"] = chunk_encoder

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

    y_train_toured_tensor = torch.tensor(y_train_toured.values, dtype=torch.float32).view(-1, 1)
    y_test_toured_tensor = torch.tensor(y_test_toured.values, dtype=torch.float32).view(-1, 1)
    y_train_applied_tensor = torch.tensor(y_train_applied.values, dtype=torch.float32).view(-1, 1)
    y_test_applied_tensor = torch.tensor(y_test_applied.values, dtype=torch.float32).view(-1, 1)
    y_train_rent_tensor = torch.tensor(y_train_rent.values, dtype=torch.float32).view(-1, 1)
    y_test_rent_tensor = torch.tensor(y_test_rent.values, dtype=torch.float32).view(-1, 1)

    # Create datasets
    train_dataset = LeadDataset(
        X_train_cat_tensor, X_train_num_tensor,
        y_train_toured_tensor, y_train_applied_tensor, y_train_rent_tensor,
        torch.tensor(train_ids.values if isinstance(train_ids, pd.Series) else train_ids)
    )

    test_dataset = LeadDataset(
        X_test_cat_tensor, X_test_num_tensor,
        y_test_toured_tensor, y_test_applied_tensor, y_test_rent_tensor,
        torch.tensor(test_ids.values if isinstance(test_ids, pd.Series) else test_ids)
    )

    # Save preprocessors for future use
    if save_preprocessors:
        if not os.path.exists(preprocessors_path):
            os.makedirs(preprocessors_path)

        preprocessors = {
            'categorical_encoders': categorical_encoders,
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

    categorical_encoders = preprocessors['categorical_encoders']
    scaler = preprocessors['scaler']
    categorical_cols = preprocessors['categorical_cols']
    numerical_cols = preprocessors['numerical_cols']

    # Extract features and targets
    X = external_data.copy()
    y_toured = (X['TOTAL_APPOINTMENT_COMPLETED'] > 0).astype(int)  # TOURED
    y_applied = (X['TOTAL_APPLIED'] > 0).astype(int)  # APPLIED
    y_rent = (X['TOTAL_RENTED'] > 0).astype(int)  # RENTED

    # Extract lead IDs for later reference
    lead_ids = X['CLIENT_PERSON_ID'] if 'CLIENT_PERSON_ID' in X.columns else np.arange(len(X))

    # Remove target columns and TOTAL_WALK_IN
    X = X.drop(['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED', 'TOTAL_WALK_IN', 'CLIENT_PERSON_ID'],
               axis=1, errors='ignore')

    # --- Temporal filtering removed ---
    # Removed filtering using LEAD_CREATED_AT

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

    # Apply the transformations for categorical features using the chunked encoders
    categorical_encoded = pd.DataFrame()

    for chunk_name, encoder in categorical_encoders.items():
        chunk_idx = int(chunk_name.split('_')[1]) if '_' in chunk_name else 0
        chunk_size = 50
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        chunk_cols = categorical_cols[start_idx:end_idx]

        if chunk_cols:
            valid_cols = [col for col in chunk_cols if col in categorical_data.columns]
            if valid_cols:
                chunk_encoded = encoder.transform(categorical_data[valid_cols])
                categorical_encoded = pd.concat([categorical_encoded, chunk_encoded], axis=1)

    # Apply transformations for numerical features
    numerical_scaled = scaler.transform(numerical_data)

    # Convert to tensors
    X_cat_tensor = torch.tensor(categorical_encoded.values, dtype=torch.float32)
    X_num_tensor = torch.tensor(numerical_scaled, dtype=torch.float32)
    y_toured_tensor = torch.tensor(y_toured.values, dtype=torch.float32).view(-1, 1)
    y_applied_tensor = torch.tensor(y_applied.values, dtype=torch.float32).view(-1, 1)
    y_rent_tensor = torch.tensor(y_rent.values, dtype=torch.float32).view(-1, 1)

    # Create dataset
    external_dataset = LeadDataset(
        X_cat_tensor, X_num_tensor,
        y_toured_tensor, y_applied_tensor, y_rent_tensor,
        torch.tensor(lead_ids.values if isinstance(lead_ids, pd.Series) else lead_ids)
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
