import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import category_encoders as ce
from tqdm import tqdm
import joblib
import os

# Import the LeadDataset from training_pipeline
from training_pipeline import LeadDataset


def preprocess_data(
        data_path,
        dict_path=None,
        dict_map_path=None,
        target_cols=['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED'],
        test_size=0.2,
        random_state=42,
        max_categories=100,  # Reduced from default to prevent too many categories
        save_preprocessors=True,
        preprocessors_path='./preprocessors',
        num_subsets=1,  # How many subsets to create (default=1 => no subsets, just entire data)
        subset_size=2000,  # Size of each subset (default 2k for initial selection)
        balance_classes=False,  # Whether to attempt balancing in each subset
        enhance_toured_features=True,  # NEW: Add interaction features for toured stage
        use_percentages=False,
        toured_pct=0.5,
        applied_pct=0.5,
        rented_pct=0.5,
        toured_k=2000,
        applied_k=2000,
        rented_k=2000
):
    """
    Preprocess the lead data for the neural network model.
    
    The model can use either fixed counts or percentage-based selection:
    - Fixed counts: Select top k leads at each stage (e.g., 2000→1000→500→250)
    - Percentage-based: Select top p% of leads at each stage based on data distribution
    - Adaptive: Automatically determine percentages based on actual rates in the data
    
    This function handles data cleaning, feature engineering, and train/test splitting.
    """
    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    
    print("\n1. Loading Data...")
    print("-"*40)
    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"* Loaded data with {len(data):,} rows and {len(data.columns):,} columns")
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {str(e)}")

    # Load data dictionary if provided
    data_dict = None
    if dict_path and os.path.exists(dict_path):
        print("\n2. Loading Data Dictionary...")
        print("-"*40)
        try:
            data_dict = pd.read_csv(dict_path)
            print(f"* Loaded dictionary with {len(data_dict):,} entries")
        except Exception as e:
            print(f"Warning: Could not load data dictionary from {dict_path}: {str(e)}")

    # Load dictionary mapping if provided
    dict_map = None
    if dict_map_path and os.path.exists(dict_map_path):
        print("Loading dictionary mapping...")
        try:
            dict_map = pd.read_csv(dict_map_path)
            print(f"Loaded dictionary mapping with {len(dict_map)} entries")
        except Exception as e:
            print(f"Warning: Could not load dictionary mapping from {dict_map_path}: {str(e)}")
            # Continue without dictionary mapping

    print("\n3. Creating Target Variables...")
    print("-"*40)
    # Check for required target columns
    for col in target_cols:
        if col not in data.columns:
            raise ValueError(f"Target column '{col}' not found in dataset.")

    y_toured = (data[target_cols[0]] > 0).astype(int)
    y_applied = (data[target_cols[1]] > 0).astype(int)
    y_rent = (data[target_cols[2]] > 0).astype(int)

    print("Target Distributions:")
    print(f"* TOURED:  {y_toured.mean() * 100:>6.2f}% positive ({y_toured.sum():,} of {len(y_toured):,})")
    print(f"* APPLIED: {y_applied.mean() * 100:>6.2f}% positive ({y_applied.sum():,} of {len(y_applied):,})")
    print(f"* RENTED:  {y_rent.mean() * 100:>6.2f}% positive ({y_rent.sum():,} of {len(y_rent):,})")

    # Calculate stage progression rates for adaptive thresholds
    stage_rates = {
        'toured_rate': y_toured.mean(),
        'applied_rate': y_applied.mean(),
        'rented_rate': y_rent.mean(),
        # Add conditional rates if possible
        'applied_given_toured': y_applied[y_toured == 1].mean() if y_toured.sum() > 0 else 0,
        'rented_given_applied': y_rent[y_applied == 1].mean() if y_applied.sum() > 0 else 0
    }
    print("\nStage Progression Rates (for adaptive thresholds):")
    print(f"* Overall toured rate: {stage_rates['toured_rate']*100:.2f}%")
    print(f"* Overall applied rate: {stage_rates['applied_rate']*100:.2f}%")
    print(f"* Overall rented rate: {stage_rates['rented_rate']*100:.2f}%")
    print(f"* Applied rate among toured leads: {stage_rates['applied_given_toured']*100:.2f}%")
    print(f"* Rented rate among applied leads: {stage_rates['rented_given_applied']*100:.2f}%")

    print("\n4. Feature Engineering...")
    print("-"*40)
    # Use data dictionary to find "important columns" if wanted
    important_cols = []
    if data_dict is not None:
        lead_keywords = ['lead', 'customer', 'client', 'property', 'apartment', 'rent', 'tour', 'visit']
        for keyword in lead_keywords:
            matching_fields = data_dict[data_dict['LONG_DESCRIPTION'].str.contains(keyword, case=False, na=False)]
            important_cols.extend(matching_fields['FIELD_NAME'].tolist())
        print(f"* Identified {len(important_cols)} potentially important columns from data dictionary")
    
    # NEW: Enhanced feature engineering for toured prediction
    if enhance_toured_features:
        print("\n4.1 Creating specialized features for toured prediction...")
        
        # Track initial column count for reporting
        initial_column_count = len(data.columns)
        
        # Try to identify date columns
        date_columns = []
        for col in data.columns:
            # Simple heuristic - column name contains 'date' or 'time'
            if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
                try:
                    # Convert to datetime and keep track of columns that work
                    pd.to_datetime(data[col], errors='coerce')
                    date_columns.append(col)
                except:
                    pass
        
        print(f"* Identified {len(date_columns)} potential date/time columns")
        
        # Create time-based features if date columns exist
        if date_columns:
            # Find most promising date column (highest correlation with toured)
            best_date_col = None
            best_corr = 0
            for col in date_columns:
                try:
                    # Convert to datetime
                    data[f'{col}_dt'] = pd.to_datetime(data[col], errors='coerce')
                    
                    # Extract day of week
                    data[f'{col}_dow'] = data[f'{col}_dt'].dt.dayofweek
                    
                    # Check correlation
                    corr = abs(data[f'{col}_dow'].corr(y_toured, method='spearman'))
                    if corr > best_corr:
                        best_corr = corr
                        best_date_col = col
                except:
                    # Skip if there are issues with this column
                    continue
                    
            if best_date_col:
                print(f"* Using '{best_date_col}' for time-based features (corr={best_corr:.4f})")
                col = best_date_col
                
                # Create more time features from the best date column
                try:
                    data[f'{col}_month'] = data[f'{col}_dt'].dt.month
                    data[f'{col}_day'] = data[f'{col}_dt'].dt.day
                    data[f'{col}_hour'] = data[f'{col}_dt'].dt.hour
                    data[f'{col}_weekend'] = (data[f'{col}_dow'] >= 5).astype(int)
                    
                    # Combine features - time of day categories
                    data['time_category'] = 0  # default
                    data.loc[data[f'{col}_hour'].between(0, 5), 'time_category'] = 1  # night
                    data.loc[data[f'{col}_hour'].between(6, 11), 'time_category'] = 2  # morning
                    data.loc[data[f'{col}_hour'].between(12, 17), 'time_category'] = 3  # afternoon
                    data.loc[data[f'{col}_hour'].between(18, 23), 'time_category'] = 4  # evening
                    
                    print(f"* Created time-of-day categories: Night (1), Morning (2), Afternoon (3), Evening (4)")
                except:
                    print(f"* Could not create all time features for {col}")
        
        # Try to identify lead quality features
        quality_keywords = ['income', 'credit', 'budget', 'qual', 'eligib', 'grade', 'tier', 'score']
        quality_cols = []
        
        for kw in quality_keywords:
            matching = [col for col in data.columns if kw in col.lower()]
            quality_cols.extend(matching)
        
        print(f"* Identified {len(quality_cols)} potential lead quality indicators")
        
        # Create lead quality score - average of z-scores from quality columns
        quality_features = []
        for col in quality_cols:
            try:
                if pd.api.types.is_numeric_dtype(data[col]):
                    # Standardize feature
                    col_mean = data[col].mean()
                    col_std = data[col].std()
                    if col_std > 0:
                        data[f'{col}_zscore'] = (data[col] - col_mean) / col_std
                        quality_features.append(f'{col}_zscore')
            except:
                continue
                
        if quality_features:
            try:
                print(f"* Creating lead quality score from {len(quality_features)} features")
                data['lead_quality_score'] = data[quality_features].mean(axis=1)
                
                # Fill missing values with median
                data['lead_quality_score'] = data['lead_quality_score'].fillna(data['lead_quality_score'].median())
            except:
                print("* Could not create lead quality score")
        
        # Create location features if zip/location information is present
        location_keywords = ['zip', 'state', 'city', 'postal', 'region', 'area', 'location', 'address']
        location_cols = []
        
        for kw in location_keywords:
            matching = [col for col in data.columns if kw.lower() in col.lower()]
            location_cols.extend(matching)
            
        print(f"* Identified {len(location_cols)} potential location columns")
        
        # Create location clusters based on zip or region
        location_cat_cols = []
        for col in location_cols:
            try:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    unique_values = data[col].nunique()
                    if 5 <= unique_values <= 1000:  # Reasonable number of categories
                        location_cat_cols.append(col)
            except:
                continue
                
        if location_cat_cols:
            # Use most granular location column (highest unique count)
            best_loc_col = max(location_cat_cols, key=lambda x: data[x].nunique())
            print(f"* Using '{best_loc_col}' for location features ({data[best_loc_col].nunique()} unique values)")
            
            # For zip/postal code, try to get first 3 digits (regional area)
            if 'zip' in best_loc_col.lower() or 'postal' in best_loc_col.lower():
                try:
                    data[f'{best_loc_col}_region'] = data[best_loc_col].astype(str).str[:3]
                    print(f"* Created region feature from {best_loc_col}")
                except:
                    pass
        
        # Create combined features (interactions)
        # Try combinations of important features
        try:
            # Day of week x time of day
            if 'time_category' in data.columns and f'{best_date_col}_dow' in data.columns:
                data['day_time_interaction'] = data[f'{best_date_col}_dow'] * 10 + data['time_category']
                print("* Created day_time_interaction feature")
                
            # Quality x Location
            if 'lead_quality_score' in data.columns and f'{best_loc_col}_region' in data.columns:
                # Group quality by region
                region_avg_quality = data.groupby(f'{best_loc_col}_region')['lead_quality_score'].mean()
                data['region_quality_diff'] = data.apply(
                    lambda row: row['lead_quality_score'] - region_avg_quality.get(row[f'{best_loc_col}_region'], 0),
                    axis=1
                )
                print("* Created region_quality_diff feature")
        except:
            print("* Could not create some interaction features")
        
        # Calculate the actual number of new columns added
        new_features_added = len(data.columns) - initial_column_count
        print(f"* Feature engineering complete - added {new_features_added} new features")

    # Drop target columns from the feature set
    print("\n5. Feature Selection...")
    print("-"*40)
    X = data.drop(target_cols, axis=1)

    # Drop specified columns
    cols_to_drop = [
        "HASH", "RECD_LUID", "RECD_P_ID", "RECD_A_ID", "CLIENT_PERSON_ID", "CLIENT_ID", "RN",
        "FNAM_FNAM", "MNAM_MNAM", "PFXT_PFXT", "SNAM_SNAM", "SFXT_SFXT", "FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE",
        "ADDRESS_LINE_1", "ADDRESS_LINE_2", "CITY", "STRT_NAME_I1", "STRT_POST_I1", "STRT_PRED_I1", "STRT_SUFX_I1",
        "QUALIFIED", "EXTRACT_DATE", "NCOA_MOVE_UPDATE_DATE", "NCOA_MOVE_UPDATE_METHOD_CODE",
        "EST_CURRENT_MORTGAGE_AMOUNT", "ENRICHMENTESTIMATED_CURRENT_MORTGAGE_AMT", "EST_MONTHLY_MORTGAGE_PAYMENT",
        "EST_CURRENT_LOAN-TO-VALUE_RATIO", "EST_AVAILABLE_EQUITY_LL", "ESTIMATED_AVAILABLE_EQUITY",
        "EQTY_LNDR_I1", "MORT_LNDR_I1", "REFI_LNDR_I1",
        "GROUP_ID", "GROUP_NAME", "LEAD_CREATED_AT", "CITY_PLAC_I1", "COUNTY_CODE_I1", "STATE_ABBR_I1", "STATE_I1",
        "RECD_ZIPC_I1", "SCDY_NUMB_I1", "SCDY_DESG_I1", "INVS_TYPE_I1", "PRCH_TYPE_I1", "TOTAL_WALK_IN"
    ]
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')

    # Remove columns with >95% missing values
    missing_pct = X.isnull().mean()
    cols_to_drop_missing = missing_pct[missing_pct > 0.95].index
    X = X.drop(cols_to_drop_missing, axis=1)
    print(f"* Dropped {len(cols_to_drop_missing)} columns with >95% missing values")

    # Separate ID column if it exists
    if 'CLIENT_PERSON_ID' in X.columns:
        lead_ids = X['CLIENT_PERSON_ID'].copy()
        X = X.drop('CLIENT_PERSON_ID', axis=1)
    else:
        lead_ids = pd.Series(np.arange(len(X)))
        print("* No CLIENT_PERSON_ID found, creating sequential IDs")

    # Identify categorical vs numerical columns - with better high-cardinality handling
    categorical_cols = []
    numerical_cols = []
    high_cardinality_cols = []
    
    for col in X.columns:
        n_unique = X[col].nunique()
        if X[col].dtype == 'object' or (X[col].dtype in ['int64', 'float64'] and n_unique < 20):
            if n_unique <= max_categories:
                categorical_cols.append(col)
            else:
                # For high-cardinality categorical columns, either:
                # 1. Treat as numerical if numeric type
                # 2. Drop if text type with too many values (would create huge embeddings)
                if X[col].dtype in ['int64', 'float64']:
                    numerical_cols.append(col)
                    print(f"* Moving high-cardinality column '{col}' ({n_unique} values) to numerical features")
                else:
                    high_cardinality_cols.append(col)
                    print(f"* Dropping high-cardinality column '{col}' ({n_unique} values) to prevent memory issues")
        elif X[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)

    # Drop high-cardinality categorical columns to prevent embedding explosion
    X = X.drop(columns=high_cardinality_cols, errors='ignore')
    
    print(f"* Identified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")
    print(f"* Dropped {len(high_cardinality_cols)} high-cardinality categorical columns to prevent memory issues")

    print("\n6. Handling Missing Values...")
    print("-"*40)
    # Handle missing values
    # For numerical: impute with median and handle infinity values
    numerical_data = X[numerical_cols].copy()

    # Check for columns with all missing values
    all_missing = numerical_data.columns[numerical_data.isnull().all()].tolist()
    if all_missing:
        print(f"Warning: {len(all_missing)} numerical columns have all missing values. Dropping: {all_missing}")
        numerical_data = numerical_data.drop(columns=all_missing)
        numerical_cols = [col for col in numerical_cols if col not in all_missing]

    # Handle infinity values
    for col in numerical_cols:
        if numerical_data[col].dtype in ['int64', 'float64']:
            # Replace inf/-inf with NaN first
            mask = np.isinf(numerical_data[col])
            if mask.any():
                num_inf = mask.sum()
                print(f"* Replacing {num_inf} infinity values in '{col}' with NaN")
                numerical_data.loc[mask, col] = np.nan
    
    # Now impute with median
    for col in numerical_cols:
        median_val = numerical_data[col].median()
        if pd.isna(median_val):
            # If median is NaN, use 0 instead
            median_val = 0
            print(f"* Column '{col}' has NaN median, using 0 for imputation")
        numerical_data[col] = numerical_data[col].fillna(median_val)
    
    # Clip extreme values to prevent training instability
    for col in numerical_cols:
        if numerical_data[col].dtype in ['float64']:
            q1 = numerical_data[col].quantile(0.01)
            q3 = numerical_data[col].quantile(0.99)
            # Wider range to preserve more data while removing extremes
            lower_bound = q1 - 5 * (q3 - q1)
            upper_bound = q3 + 5 * (q3 - q1)
            
            # Count and clip outliers
            outliers = ((numerical_data[col] < lower_bound) | (numerical_data[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"* Clipping {outliers} extreme values in '{col}'")
                numerical_data[col] = numerical_data[col].clip(lower_bound, upper_bound)

    # For categorical: impute with mode
    categorical_data = X[categorical_cols].copy()
    for col in categorical_cols:
        # Check if column is all missing
        if categorical_data[col].isnull().all():
            print(f"Warning: Categorical column '{col}' has all missing values. Using default value 'missing'")
            categorical_data[col] = 'missing'
        else:
            # Get the mode, but default to 'missing' if there's an issue
            try:
                mode_val = categorical_data[col].mode()[0]
                categorical_data[col] = categorical_data[col].fillna(mode_val)
            except:
                print(f"* Error finding mode for '{col}', using 'missing' value")
                categorical_data[col] = categorical_data[col].fillna('missing')

    # ---------- HELPER FUNCTION FOR BUILDING A SINGLE (train_dataset, test_dataset) -----------
    def build_dataset_splits(cat_data, num_data, y_tour, y_app, y_rent, lead_id_array):
        """
        Given a portion of the data (cat_data, num_data, labels, lead_ids),
        split into train/test and do the target encoding, scaling, etc.
        Returns train_dataset, test_dataset, plus dimension info.
        """
        # Ensure data is valid
        if len(cat_data) == 0 and len(num_data) == 0:
            raise ValueError("No features available after preprocessing")

        if len(cat_data) != len(num_data) or len(cat_data) != len(y_tour):
            raise ValueError(
                f"Mismatched lengths: cat_data={len(cat_data)}, num_data={len(num_data)}, labels={len(y_tour)}")

        # Split - use stratify only if we have both classes
        stratify = y_rent if y_rent.nunique() > 1 else None
        if stratify is None:
            print("Warning: Cannot stratify split - target 'rented' has only one class")

        X_train_cat, X_test_cat, X_train_num, X_test_num, \
            y_train_toured, y_test_toured, \
            y_train_applied, y_test_applied, \
            y_train_rent, y_test_rent, \
            train_ids, test_ids = train_test_split(
            cat_data, num_data, y_tour, y_app, y_rent, lead_id_array,
            test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Convert everything to dataframes, so we can chunk-encode
        X_train_cat = pd.DataFrame(X_train_cat, columns=cat_data.columns)
        X_test_cat = pd.DataFrame(X_test_cat, columns=cat_data.columns)

        # Start target encoding in chunks
        chunk_size = 50
        X_train_cat_encoded = pd.DataFrame()
        X_test_cat_encoded = pd.DataFrame()
        categorical_encoders = {}
        categorical_dims = []  # List to store categorical dimensions (number of unique values + 1)

        # Use label encoding instead of target encoding for categorical features
        for i in range(0, len(categorical_cols), chunk_size):
            chunk_cols = categorical_cols[i: i + chunk_size]
            # Filter to only include columns actually in the data
            chunk_cols = [col for col in chunk_cols if col in X_train_cat.columns]

            if not chunk_cols:
                continue  # Skip if no columns in this chunk

            # Create a dictionary to store label encoders for each column
            label_encoders = {}
            chunk_train_encoded = pd.DataFrame()
            chunk_test_encoded = pd.DataFrame()
            
            for col in chunk_cols:
                try:
                    # Create a label encoder for this column
                    label_encoder = LabelEncoder()
                    # Ensure values are strings and handle special characters
                    combined_values = pd.concat([X_train_cat[col], X_test_cat[col]]).astype(str)
                    
                    # Limit the number of unique categories to prevent memory issues
                    if len(combined_values.unique()) > 1000:
                        print(f"Warning: Column {col} has {len(combined_values.unique())} unique values. " +
                              f"Limiting to top 1000 most frequent.")
                        value_counts = combined_values.value_counts().nlargest(999)
                        top_values = set(value_counts.index)
                        # Replace rare values with 'rare'
                        combined_values = combined_values.apply(lambda x: x if x in top_values else 'rare')
                        
                    label_encoder.fit(combined_values)
                    
                    # Transform train and test data
                    train_encoded = label_encoder.transform(X_train_cat[col].astype(str))
                    test_encoded = label_encoder.transform(X_test_cat[col].astype(str))
                    
                    # Store transformed data
                    chunk_train_encoded[col] = train_encoded
                    chunk_test_encoded[col] = test_encoded
                    
                    # Store the label encoder
                    label_encoders[col] = label_encoder
                    
                    # Store the categorical dimension (number of unique categories)
                    categorical_dims.append(len(label_encoder.classes_))
                except Exception as e:
                    print(f"Warning: Error encoding column {col}: {str(e)}")
                    # If encoding fails, use a placeholder with 2 categories (0 and 1)
                    chunk_train_encoded[col] = 0
                    chunk_test_encoded[col] = 0
                    categorical_dims.append(2)

            X_train_cat_encoded = pd.concat([X_train_cat_encoded, chunk_train_encoded], axis=1)
            X_test_cat_encoded = pd.concat([X_test_cat_encoded, chunk_test_encoded], axis=1)
            categorical_encoders[f"chunk_{i // chunk_size}"] = label_encoders

        # Scale numerical
        if X_train_num.shape[1] > 0:
            scaler = StandardScaler()
            
            # Create a copy of data for safer manipulation
            X_train_num_safe = X_train_num.copy()
            X_test_num_safe = X_test_num.copy()
            
            # Replace any remaining inf values with large finite numbers
            X_train_num_safe = np.nan_to_num(X_train_num_safe, nan=0, posinf=1e6, neginf=-1e6)
            X_test_num_safe = np.nan_to_num(X_test_num_safe, nan=0, posinf=1e6, neginf=-1e6)
            
            # Fit and transform with robust handling
            try:
                X_train_num_scaled = scaler.fit_transform(X_train_num_safe)
                X_test_num_scaled = scaler.transform(X_test_num_safe)
                
                # Additional post-scaling outlier clipping for stability
                # Clip to reasonable range to prevent extreme values
                X_train_num_scaled = np.clip(X_train_num_scaled, -10, 10)
                X_test_num_scaled = np.clip(X_test_num_scaled, -10, 10)
            except Exception as e:
                print(f"Warning: Scaling error: {str(e)}, using simple standardization")
                # Fallback to simpler standardization
                means = np.nanmean(X_train_num_safe, axis=0)
                stds = np.nanstd(X_train_num_safe, axis=0)
                stds[stds == 0] = 1  # Prevent division by zero
                
                X_train_num_scaled = (X_train_num_safe - means) / stds
                X_test_num_scaled = (X_test_num_safe - means) / stds
                
                # Clip values for stability
                X_train_num_scaled = np.clip(X_train_num_scaled, -10, 10)
                X_test_num_scaled = np.clip(X_test_num_scaled, -10, 10)
        else:
            print("Warning: No numerical features available")
            # Create empty array with correct shape
            X_train_num_scaled = np.zeros((len(X_train_num), 0))
            X_test_num_scaled = np.zeros((len(X_test_num), 0))
            scaler = None

        # Convert to tensors with proper dtype specification and validation
        try:
            # Ensure categorical data is integer for embedding lookup
            X_train_cat_tensor = torch.tensor(X_train_cat_encoded.values, dtype=torch.long)
            X_test_cat_tensor = torch.tensor(X_test_cat_encoded.values, dtype=torch.long)
            
            # Ensure numerical data is float
            X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
            X_test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32)
            
            # Handle labels - ensure they're clean floats between 0-1
            y_train_toured_np = np.clip(y_train_toured.values if isinstance(y_train_toured, pd.Series) else y_train_toured, 0, 1)
            y_test_toured_np = np.clip(y_test_toured.values if isinstance(y_test_toured, pd.Series) else y_test_toured, 0, 1)
            
            y_train_applied_np = np.clip(y_train_applied.values if isinstance(y_train_applied, pd.Series) else y_train_applied, 0, 1)
            y_test_applied_np = np.clip(y_test_applied.values if isinstance(y_test_applied, pd.Series) else y_test_applied, 0, 1)
            
            y_train_rent_np = np.clip(y_train_rent.values if isinstance(y_train_rent, pd.Series) else y_train_rent, 0, 1)
            y_test_rent_np = np.clip(y_test_rent.values if isinstance(y_test_rent, pd.Series) else y_test_rent, 0, 1)
            
            # Convert to tensors with proper shape
            y_train_toured_tensor = torch.tensor(y_train_toured_np, dtype=torch.float32).view(-1, 1)
            y_test_toured_tensor = torch.tensor(y_test_toured_np, dtype=torch.float32).view(-1, 1)
            
            y_train_applied_tensor = torch.tensor(y_train_applied_np, dtype=torch.float32).view(-1, 1)
            y_test_applied_tensor = torch.tensor(y_test_applied_np, dtype=torch.float32).view(-1, 1)
            
            y_train_rent_tensor = torch.tensor(y_train_rent_np, dtype=torch.float32).view(-1, 1)
            y_test_rent_tensor = torch.tensor(y_test_rent_np, dtype=torch.float32).view(-1, 1)
            
            # Prepare lead IDs
            train_ids_tensor = torch.tensor(train_ids.values if isinstance(train_ids, pd.Series) else train_ids)
            test_ids_tensor = torch.tensor(test_ids.values if isinstance(test_ids, pd.Series) else test_ids)
            
        except Exception as e:
            print(f"Error creating tensors: {str(e)}")
            raise

        # Create dataset objects
        train_dataset = LeadDataset(
            X_train_cat_tensor, X_train_num_tensor,
            y_train_toured_tensor, y_train_applied_tensor, y_train_rent_tensor,
            train_ids_tensor
        )
        test_dataset = LeadDataset(
            X_test_cat_tensor, X_test_num_tensor,
            y_test_toured_tensor, y_test_applied_tensor, y_test_rent_tensor,
            test_ids_tensor
        )

        # Print dataset statistics for debugging
        print(f"Created datasets - Train: {len(train_dataset)} examples, Test: {len(test_dataset)} examples")
        print(f"Categorical features: {X_train_cat_tensor.shape[1]}, Numerical features: {X_train_num_tensor.shape[1]}")
        
        numerical_dim = X_train_num_scaled.shape[1]
        feature_names = list(X_train_cat_encoded.columns) + list(num_data.columns)

        return train_dataset, test_dataset, categorical_encoders, scaler, categorical_dims, numerical_dim, feature_names

    # -------------- If num_subsets == 1, just do the entire dataset --------------
    if num_subsets <= 1:
        print("\n7. Train/Test Split...")
        print("-"*40)
        if enhance_toured_features:
            print("Enhanced feature engineering enabled for toured prediction")
            
        # Update this to show the correct flow based on percentage or fixed count selection
        if use_percentages:
            print(f"Using percentage-based selection: {toured_pct*100:.1f}% -> {applied_pct*100:.1f}% -> {rented_pct*100:.1f}%")
        else:
            print(f"Cascade Flow: {toured_k} leads -> {applied_k} toured -> {rented_k} applied -> rented")
            
        torch.multiprocessing.set_sharing_strategy('file_system')

        try:
            train_dataset, test_dataset, cat_encoders, scaler_obj, cat_dims, num_dim, feat_names = build_dataset_splits(
                categorical_data, numerical_data, y_toured, y_applied, y_rent, lead_ids
            )

            print(f"* Training set: {len(train_dataset):,} samples")
            print(f"* Testing set:  {len(test_dataset):,} samples")

            print("\n8. Final Class Distribution (Train Set):")
            print("-"*40)
            toured_labels = torch.stack([item[2] for item in train_dataset])
            applied_labels = torch.stack([item[3] for item in train_dataset])
            rented_labels = torch.stack([item[4] for item in train_dataset])
            
            print(f"* TOURED:  {torch.mean(toured_labels).item() * 100:>6.2f}% positive")
            print(f"* APPLIED: {torch.mean(applied_labels).item() * 100:>6.2f}% positive")
            print(f"* RENTED:  {torch.mean(rented_labels).item() * 100:>6.2f}% positive")
            print("\n" + "="*80 + "\n")

            # Save preprocessors if requested
            if save_preprocessors:
                if not os.path.exists(preprocessors_path):
                    os.makedirs(preprocessors_path)
                # We'll just store the single-run encoders/scaler
                preprocessors = {
                    'categorical_encoders': cat_encoders,
                    'scaler': scaler_obj,
                    'categorical_cols': categorical_cols,
                    'numerical_cols': numerical_cols,
                    'feature_names': feat_names,
                    'stage_rates': stage_rates
                }

                # Use joblib instead of pickle
                joblib.dump(preprocessors, os.path.join(preprocessors_path, 'preprocessors.joblib'))
                print(f"Saved preprocessors to {os.path.join(preprocessors_path, 'preprocessors.joblib')}")

            return train_dataset, test_dataset, cat_dims, num_dim, feat_names, stage_rates

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    # -------------- Otherwise, build multiple subsets --------------
    else:
        print(f"Will create {num_subsets} subsets, each of size {subset_size} (balance={balance_classes})")
        # We don't do a single "train/test" on the entire data; instead,
        # we create multiple smaller runs.

        # Convert main data to a single DataFrame for easy sampling
        big_df = pd.DataFrame({
            "y_toured": y_toured,
            "y_applied": y_applied,
            "y_rent": y_rent,
            "lead_id": lead_ids
        })
        # Add back in the cat & num data
        # We'll keep them separate for processing, but let's unify indexes
        cat_df = categorical_data.reset_index(drop=True)
        num_df = numerical_data.reset_index(drop=True)
        big_df = big_df.reset_index(drop=True)
        big_df_catnum = pd.concat([big_df, cat_df, num_df], axis=1)

        # If balancing is requested, we attempt to select an equal number of rented=1 and rented=0
        # in each subset. Or at least keep the ratio 50-50 if possible.
        # Then we do train_test_split on that subset, do the transforms, etc.
        subsets_list = []

        rng = np.random.RandomState(random_state)  # for reproducible sampling
        rented_1_df = big_df_catnum[big_df_catnum["y_rent"] == 1]
        rented_0_df = big_df_catnum[big_df_catnum["y_rent"] == 0]

        print(f"Found {len(rented_1_df)} positive and {len(rented_0_df)} negative rental examples")

        for subset_i in range(num_subsets):
            try:
                if balance_classes:
                    # Modified: Calculate target number of rented=1 examples (target: 12.5%)
                    target_rented_count = int(subset_size * 0.125)  # 250 out of 2000
                    sample_1 = rented_1_df.sample(n=min(target_rented_count, len(rented_1_df)),
                                                  replace=False,
                                                  random_state=rng.randint(0, 999999))
                    sample_0 = rented_0_df.sample(n=min(subset_size - len(sample_1), len(rented_0_df)),
                                                  replace=False,
                                                  random_state=rng.randint(0, 999999))
                    subset_df = pd.concat([sample_1, sample_0], axis=0)
                else:
                    # just random from entire big_df_catnum
                    subset_df = big_df_catnum.sample(n=min(subset_size, len(big_df_catnum)), replace=False,
                                                     random_state=rng.randint(0, 999999))

                # Now we re-split out labels, cat_data, num_data, lead_ids
                y_tour_sub = subset_df["y_toured"].values
                y_app_sub = subset_df["y_applied"].values
                y_rent_sub = subset_df["y_rent"].values
                lead_id_sub = subset_df["lead_id"].values

                # Rebuild the cat & num data from the subset
                subset_cat = subset_df[cat_df.columns]
                subset_num = subset_df[num_df.columns]

                print(f"\nBuilding subset {subset_i + 1}/{num_subsets} with {len(subset_df)} leads")
                print(
                    f"  Rented distribution: {np.mean(y_rent_sub) * 100:.2f}% positive ({np.sum(y_rent_sub)} of {len(y_rent_sub)})")

                torch.multiprocessing.set_sharing_strategy('file_system')
                tr_ds, ts_ds, cat_enc, scaler_obj, cat_dims, num_dim, feat_names = build_dataset_splits(
                    subset_cat, subset_num, y_tour_sub, y_app_sub, y_rent_sub, lead_id_sub
                )
                # We'll store them
                subsets_list.append((tr_ds, ts_ds))

            except Exception as e:
                print(f"Error building subset {subset_i + 1}: {str(e)}")
                print("Skipping this subset and continuing")
                continue

        # For multiple subsets, we typically won't store *one* set of preprocessors
        # because each subset can have slightly different encoders/scalers.
        # If you want to store *each* subset's encoders, you'll want to do so in a loop.
        # Here, we'll just store the final one by default:
        if save_preprocessors and len(subsets_list) > 0:
            if not os.path.exists(preprocessors_path):
                os.makedirs(preprocessors_path)
            # Save the last subset's preprocessors, or you can store them in a list
            preprocessors = {
                'categorical_encoders': cat_enc,
                'scaler': scaler_obj,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols,
                'feature_names': feat_names,
                'stage_rates': stage_rates  # Include stage rates
            }

            # Use joblib instead of pickle
            joblib.dump(preprocessors, os.path.join(preprocessors_path, 'preprocessors.joblib'))
            print(f"Saved preprocessors to {os.path.join(preprocessors_path, 'preprocessors.joblib')}")

        if len(subsets_list) == 0:
            raise ValueError("Failed to create any valid subsets")

        # Return a dictionary describing everything
        # The user can iterate over the subsets for training
        return {
            'subsets': subsets_list,
            'categorical_dims': cat_dims,
            'numerical_dim': num_dim,
            'feature_names': feat_names,
            'stage_rates': stage_rates  # Include stage rates
        }


def prepare_external_examples(external_data, preprocessors_path='./preprocessors'):
    """
    Prepare external examples for fine-tuning using saved preprocessors

    Args:
        external_data: DataFrame containing the external examples
        preprocessors_path: Path to the saved preprocessors

    Returns:
        external_dataset: Dataset containing processed external examples
    """
    # Determine which preprocessor file to use
    preprocessor_file = os.path.join(preprocessors_path, 'preprocessors.joblib')
    if not os.path.exists(preprocessor_file):
        # Try the subset last file
        preprocessor_file = os.path.join(preprocessors_path, 'preprocessors_subset_last.joblib')
        if not os.path.exists(preprocessor_file):
            raise FileNotFoundError(f"No preprocessor files found in {preprocessors_path}")

    print(f"Loading preprocessors from {preprocessor_file}")

    # Load with joblib instead of pickle
    try:
        preprocessors = joblib.load(preprocessor_file)
    except Exception as e:
        raise ValueError(f"Error loading preprocessors: {str(e)}")

    categorical_encoders = preprocessors['categorical_encoders']
    scaler = preprocessors['scaler']
    categorical_cols = preprocessors['categorical_cols']
    numerical_cols = preprocessors['numerical_cols']

    # Extract features + targets
    X = external_data.copy()

    # Check for required target columns
    target_cols = ['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED']
    for col in target_cols:
        if col not in X.columns:
            raise ValueError(f"Required target column '{col}' not found in external data")

    y_toured = (X['TOTAL_APPOINTMENT_COMPLETED'] > 0).astype(int)
    y_applied = (X['TOTAL_APPLIED'] > 0).astype(int)
    y_rent = (X['TOTAL_RENTED'] > 0).astype(int)

    print(f"External data: {len(X)} rows")
    print(f"  Toured: {y_toured.mean() * 100:.2f}% positive ({y_toured.sum()} of {len(y_toured)})")
    print(f"  Applied: {y_applied.mean() * 100:.2f}% positive ({y_applied.sum()} of {len(y_applied)})")
    print(f"  Rented: {y_rent.mean() * 100:.2f}% positive ({y_rent.sum()} of {len(y_rent)})")

    # IDs
    if 'CLIENT_PERSON_ID' in X.columns:
        lead_ids = X['CLIENT_PERSON_ID'].values
    else:
        lead_ids = np.arange(len(X))
        print("No CLIENT_PERSON_ID found, creating sequential IDs")

    # Remove target columns (and TOTAL_WALK_IN if it exists)
    drop_cols = ['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED', 'TOTAL_WALK_IN', 'CLIENT_PERSON_ID']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

    # Ensure missing columns are handled
    for col in categorical_cols:
        if col not in X.columns:
            print(f"Warning: Categorical column '{col}' not found in external data. Adding with value 'missing'")
            X[col] = 'missing'
    for col in numerical_cols:
        if col not in X.columns:
            print(f"Warning: Numerical column '{col}' not found in external data. Adding with value 0")
            X[col] = 0

    # Separate cat and num
    cat_data = X[categorical_cols].copy()
    num_data = X[numerical_cols].copy()

    # Impute missing
    for col in categorical_cols:
        cat_data[col] = cat_data[col].fillna('missing')
    for col in numerical_cols:
        num_data[col] = num_data[col].fillna(0)

    # Apply label encoders from preprocessors
    print("Applying categorical encoders...")
    cat_encoded = pd.DataFrame()

    for chunk_name, encoder_dict in categorical_encoders.items():
        try:
            # Get the columns in this chunk
            if chunk_name.startswith('chunk_'):
                chunk_idx = int(chunk_name.split('_')[1]) if '_' in chunk_name else 0
                chunk_size = 50
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_size
                chunk_cols = categorical_cols[start_idx:end_idx]
                chunk_cols = [col for col in chunk_cols if col in cat_data.columns]
                
                # For each column in this chunk, apply the corresponding label encoder
                chunk_encoded = pd.DataFrame()
                for col in chunk_cols:
                    if col in encoder_dict:
                        label_encoder = encoder_dict[col]
                        # Ensure all values are strings for encoding
                        cat_col_data = cat_data[col].astype(str)
                        
                        # Handle unseen categories
                        unique_values = set(cat_col_data.unique())
                        known_values = set(label_encoder.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            print(f"Warning: Column {col} has {len(unknown_values)} new categories not seen in training")
                            # Replace unknown values with a known value (first class)
                            for val in unknown_values:
                                cat_col_data = cat_col_data.replace(val, label_encoder.classes_[0])
                        
                        # Transform and add to encoded data
                        try:
                            encoded_values = label_encoder.transform(cat_col_data)
                            chunk_encoded[col] = encoded_values
                        except Exception as e:
                            print(f"Error encoding column {col}: {str(e)} - using zeros")
                            chunk_encoded[col] = 0
                    else:
                        print(f"Warning: No encoder found for column {col} - using zeros")
                        chunk_encoded[col] = 0
                
                cat_encoded = pd.concat([cat_encoded, chunk_encoded], axis=1)
            elif chunk_name.startswith('col_'):
                # Handle individual column encoders if any
                col = chunk_name.split('_', 1)[1]
                if col in cat_data.columns:
                    try:
                        encoded_values = encoder_dict.transform(cat_data[[col]])
                        cat_encoded[col] = encoded_values
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)} - using zeros")
                        cat_encoded[col] = 0
        except Exception as e:
            print(f"Warning: Error applying encoders for {chunk_name}: {str(e)}")

    # Scale numeric
    print("Scaling numerical features...")
    if scaler is not None and num_data.shape[1] > 0:
        try:
            num_scaled = scaler.transform(num_data)
        except Exception as e:
            print(f"Warning: Error scaling numerical data: {str(e)}")
            # Fallback to standard scaling or no scaling
            num_scaled = (num_data - num_data.mean()) / num_data.std().replace(0, 1)
    else:
        print("No scaler available or no numerical features - using standardized values")
        num_scaled = (num_data - num_data.mean()) / num_data.std().replace(0, 1)

    # Convert to tensors - categorical as long for embedding lookup
    print("Converting to tensors...")
    cat_tensor = torch.tensor(cat_encoded.values, dtype=torch.long)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
    y_toured_t = torch.tensor(y_toured.values, dtype=torch.float32).view(-1, 1)
    y_applied_t = torch.tensor(y_applied.values, dtype=torch.float32).view(-1, 1)
    y_rent_t = torch.tensor(y_rent.values, dtype=torch.float32).view(-1, 1)

    # Build dataset
    dataset = LeadDataset(
        cat_tensor, num_tensor,
        y_toured_t, y_applied_t, y_rent_t,
        torch.tensor(lead_ids)
    )

    print(f"Created external dataset with {len(dataset)} examples")
    return dataset


def visualize_group_differences(differences, stage, save_path=None):
    """
    Visualize the differences between selected and non-selected leads for a stage
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [item[0] for item in differences]
    abs_diffs = [item[1]['abs_diff'] for item in differences]

    colors = ['green' if diff > 0 else 'red' for diff in abs_diffs]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(features, abs_diffs, color=colors)
    plt.xlabel('Absolute Difference (Selected - Non-Selected)')
    plt.title(f'Top Feature Differences for {stage.capitalize()} Stage')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width >= 0:
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", ha='left', va='center')
        else:
            plt.text(width - 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", ha='right', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def train_with_seed(args, seed=None):
    # ... existing code ...
    
    # Print key configuration parameters
    print(f"\nConfiguration:")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Output file: {args.output_file}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accum}")
    
    # Only show the selection method being used
    if args.use_percentages:
        print(f"  - Using percentage-based selection")
        if args.adapt_to_data:
            print(f"  - Adapting percentages to data distribution")
        else:
            print(f"  - Percentages: {args.toured_pct*100:.1f}% -> {args.applied_pct*100:.1f}% -> {args.rented_pct*100:.1f}%")
    else:
        print(f"  - Using fixed count selection: {args.toured_k} -> {args.applied_k} -> {args.rented_k}")
        
    print(f"  - Enhanced toured features: {args.enhance_toured_features}")