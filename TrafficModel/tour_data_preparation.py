import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import joblib
import os
from tqdm import tqdm

# Assuming LeadDataset is available and can handle potentially fewer targets
# If not, adjust the LeadDataset import and instantiation accordingly
# from training_pipeline import LeadDataset

# Placeholder LeadDataset if not imported - ADJUST AS NEEDED
class LeadDataset(Dataset):
    """Minimal placeholder for torch Dataset."""
    def __init__(self, cat_features, num_features, y_toured, y_applied=None, y_rent=None, lead_ids=None):
        self.cat_features = cat_features
        self.num_features = num_features
        self.y_toured = y_toured
        # Handle potentially missing targets by creating dummy tensors if None
        self.y_applied = y_applied if y_applied is not None else torch.zeros_like(y_toured)
        self.y_rent = y_rent if y_rent is not None else torch.zeros_like(y_toured)
        self.lead_ids = lead_ids if lead_ids is not None else torch.arange(len(y_toured))

    def __len__(self):
        return len(self.y_toured)

    def __getitem__(self, idx):
        # Return consistent tuple structure, including dummy targets if needed
        return (
            self.cat_features[idx] if self.cat_features.shape[1] > 0 else torch.tensor([]), # Handle case with no cat features
            self.num_features[idx] if self.num_features.shape[1] > 0 else torch.tensor([]), # Handle case with no num features
            self.y_toured[idx],
            self.y_applied[idx],
            self.y_rent[idx],
            self.lead_ids[idx]
        )


def prepare_tour_data(
        data_path,
        dict_path=None,
        dict_map_path=None,
        target_col='TOTAL_APPOINTMENT_COMPLETED', # Default target for toured
        enhance_features=True,
        test_size=0.2,
        random_state=42,
        max_categories=250,
        save_preprocessors=True,
        preprocessors_path='./preprocessors/tour', # Specific path for tour preprocessors
):
    """
    Prepares data specifically for the toured prediction model.
    Focuses on feature engineering relevant to tour prediction and handles
    only the 'toured' target variable for processing, while maintaining
    compatibility with a LeadDataset expecting multiple targets (using dummies).
    """
    print("\n" + "="*80)
    print("TOUR DATA PREPARATION PIPELINE")
    print("="*80)

    # 1. Load Data
    print("\n1. Loading Data...")
    print("-"*40)
    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"* Loaded data with {len(data):,} rows and {len(data.columns):,} columns")
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {str(e)}")

    # Optional: Load dictionaries (can add logic if needed, similar to original)
    if dict_path and os.path.exists(dict_path): print(f"* Dictionary path provided: {dict_path} (logic not fully implemented)")
    if dict_map_path and os.path.exists(dict_map_path): print(f"* Dict map path provided: {dict_map_path} (logic not fully implemented)")


    # 2. Create Target Variable
    print("\n2. Creating Target Variable...")
    print("-"*40)
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    y_toured = (data[target_col] > 0).astype(int)
    print(f"* Target (Toured): {y_toured.mean() * 100:>6.2f}% positive ({y_toured.sum():,} of {len(y_toured):,})")

    # 3. Feature Selection & Early Cleaning
    print("\n3. Feature Selection & Early Cleaning...")
    print("-"*40)

    # --- PREVENT LEAKAGE: Drop target column *before* feature engineering ---
    # Also drop other potential future targets if they exist
    potential_future_targets = ['TOTAL_APPLIED', 'TOTAL_RENTED']
    cols_to_drop_targets = [target_col] + [col for col in potential_future_targets if col in data.columns]
    print(f"* Dropping target/future target columns: {cols_to_drop_targets}")
    data = data.drop(columns=cols_to_drop_targets, errors='ignore')

    # --- Define and drop irrelevant/leaky columns BEFORE feature engineering ---
    # (Using the same list as the original preprocess_data)
    cols_to_drop_irrelevant = [
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

    # Separate ID column *before* dropping it
    id_col_name = 'CLIENT_PERSON_ID' # Assuming this is the ID column
    if id_col_name in data.columns:
        lead_ids = data[id_col_name].copy()
        print(f"* Extracted {len(lead_ids)} {id_col_name}s")
    else:
        lead_ids = pd.Series(np.arange(len(data)), name='generated_id')
        print(f"* No {id_col_name} found, creating sequential IDs")
        id_col_name = 'generated_id' # Use the generated name

    # Drop the specified columns (including the ID column now)
    num_cols_before_drop = len(data.columns)
    all_cols_to_drop = cols_to_drop_irrelevant + [id_col_name] # Add ID col to drop list
    data = data.drop(columns=[col for col in all_cols_to_drop if col in data.columns], errors='ignore')
    num_cols_after_drop = len(data.columns)
    print(f"* Dropped {num_cols_before_drop - num_cols_after_drop} specified irrelevant/leaky columns")

    # Remove columns with >95% missing values BEFORE feature engineering
    num_cols_before_missing = len(data.columns)
    missing_pct = data.isnull().mean()
    cols_to_drop_missing = missing_pct[missing_pct > 0.95].index
    if len(cols_to_drop_missing) > 0:
        data = data.drop(cols_to_drop_missing, axis=1)
        print(f"* Dropped {len(cols_to_drop_missing)} columns with >95% missing values: {cols_to_drop_missing.tolist()}")
    else:
        print("* No columns found with >95% missing values")

    # 4. Feature Engineering (Tour Specific)
    print("\n4. Feature Engineering...")
    print("-"*40)
    if enhance_features:
        print("* Applying enhanced feature engineering for tour prediction...")
        # Create a temporary DataFrame to store new features
        new_features_df = pd.DataFrame(index=data.index)
        initial_column_count = len(data.columns)

        # Find date columns
        date_columns = []
        for col in data.columns:
            # Relaxed check for date-like names
            if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'created', 'updated']):
                try:
                    # Quick check if conversion is feasible for a sample
                    sample = data[col].dropna().iloc[:10]
                    if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().any():
                        # Check if column has enough unique values
                        if data[col].nunique() > 1:  # Avoid constant columns
                            date_columns.append(col)
                except Exception:
                    pass # Ignore columns that fail conversion or are empty
        print(f"* Identified {len(date_columns)} potential date/time columns: {date_columns}")

        best_date_col = None
        best_dt_series = None
        if date_columns:
            best_corr = -1 # Use -1 to ensure any valid correlation is chosen
            temp_y_toured_series = pd.Series(y_toured, index=data.index) # Align index for correlation

            for col in date_columns:
                try:
                    # Convert the whole column, coercing errors
                    dt_col = pd.to_datetime(data[col], errors='coerce')
                    if dt_col.isnull().all(): # Skip if conversion failed completely
                         print(f"  - Skipping date column '{col}' as conversion resulted in all NaT.")
                         continue

                    # Calculate multiple correlation metrics
                    dow_col = dt_col.dt.dayofweek
                    month_col = dt_col.dt.month
                    hour_col = dt_col.dt.hour
                    
                    # Calculate correlations for different time components
                    dow_corr = abs(dow_col.reindex(temp_y_toured_series.index).corr(temp_y_toured_series, method='spearman'))
                    month_corr = abs(month_col.reindex(temp_y_toured_series.index).corr(temp_y_toured_series, method='spearman'))
                    hour_corr = abs(hour_col.reindex(temp_y_toured_series.index).corr(temp_y_toured_series, method='spearman'))
                    
                    # Use the maximum correlation across time components
                    max_corr = max(dow_corr, month_corr, hour_corr)
                    
                    if not pd.isna(max_corr) and max_corr > best_corr:
                        best_corr = max_corr
                        best_date_col = col
                        best_dt_series = dt_col
                        print(f"  - Found new best date column: '{col}' (Max Spearman corr={max_corr:.4f})")

                except Exception as e:
                    print(f"  - Skipping date column '{col}' due to error during correlation check: {e}")
                    continue

            if best_date_col:
                print(f"* Using '{best_date_col}' for time-based features (Best Spearman corr={best_corr:.4f})")
                # Store base date features - Fill NaNs after calculations where needed
                best_dow_series = best_dt_series.dt.dayofweek # Contains NaNs
                new_features_df[f'{best_date_col}_dow'] = best_dow_series # Keep NaNs for now

                # Create more sophisticated time features
                try:
                    # Basic time features
                    new_features_df[f'{best_date_col}_month'] = best_dt_series.dt.month.fillna(-1).astype(int)
                    new_features_df[f'{best_date_col}_day'] = best_dt_series.dt.day.fillna(-1).astype(int)
                    new_features_df[f'{best_date_col}_hour'] = best_dt_series.dt.hour.fillna(-1).astype(int)
                    new_features_df[f'{best_date_col}_weekend'] = (best_dow_series >= 5).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_quarter'] = best_dt_series.dt.quarter.fillna(-1).astype(int)
                    
                    # Advanced time features
                    new_features_df[f'{best_date_col}_is_month_start'] = (best_dt_series.dt.day == 1).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_month_end'] = (best_dt_series.dt.day == best_dt_series.dt.days_in_month).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_quarter_start'] = ((best_dt_series.dt.month % 3 == 1) & (best_dt_series.dt.day == 1)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_quarter_end'] = ((best_dt_series.dt.month % 3 == 0) & (best_dt_series.dt.day == best_dt_series.dt.days_in_month)).fillna(0).astype(int)
                    
                    # Time of day features
                    new_features_df[f'{best_date_col}_is_morning'] = ((best_dt_series.dt.hour >= 6) & (best_dt_series.dt.hour < 12)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_afternoon'] = ((best_dt_series.dt.hour >= 12) & (best_dt_series.dt.hour < 18)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_evening'] = ((best_dt_series.dt.hour >= 18) & (best_dt_series.dt.hour < 22)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_night'] = ((best_dt_series.dt.hour >= 22) | (best_dt_series.dt.hour < 6)).fillna(0).astype(int)
                    
                    # Season features
                    new_features_df[f'{best_date_col}_is_winter'] = ((best_dt_series.dt.month >= 12) | (best_dt_series.dt.month <= 2)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_spring'] = ((best_dt_series.dt.month >= 3) & (best_dt_series.dt.month <= 5)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_summer'] = ((best_dt_series.dt.month >= 6) & (best_dt_series.dt.month <= 8)).fillna(0).astype(int)
                    new_features_df[f'{best_date_col}_is_fall'] = ((best_dt_series.dt.month >= 9) & (best_dt_series.dt.month <= 11)).fillna(0).astype(int)
                    
                    # Time since features
                    current_date = pd.Timestamp.now()
                    new_features_df[f'{best_date_col}_days_since'] = (current_date - best_dt_series).dt.days.fillna(-1)
                    new_features_df[f'{best_date_col}_weeks_since'] = (new_features_df[f'{best_date_col}_days_since'] / 7).fillna(-1)
                    new_features_df[f'{best_date_col}_months_since'] = (new_features_df[f'{best_date_col}_days_since'] / 30).fillna(-1)
                    
                    print(f"* Created comprehensive time features from '{best_date_col}'")

                except Exception as e:
                    print(f"* Warning: Could not create all time features for {best_date_col}: {str(e)}")
            else:
                 print("* No suitable date column found or selected for detailed time features.")

        # Engagement Features with enhanced normalization
        engagement_keywords = ['click', 'visit', 'view', 'open', 'engage', 'activity', 'interest', 'action', 'contact', 'session', 'page', 'time_on']
        engagement_cols = []
        for kw in engagement_keywords:
            # Ensure column exists and is numeric before adding
            matching = [col for col in data.columns if kw.lower() in col.lower() and pd.api.types.is_numeric_dtype(data[col])]
            engagement_cols.extend(matching)
        engagement_cols = sorted(list(set(engagement_cols))) # Unique & sorted
        print(f"* Identified {len(engagement_cols)} potential numeric engagement columns: {engagement_cols}")

        behavior_features = []
        if engagement_cols:
            try:
                for col in engagement_cols:
                    # Enhanced normalization with robust scaling
                    col_filled = data[col].fillna(0)
                    col_median = col_filled.median()
                    col_iqr = col_filled.quantile(0.75) - col_filled.quantile(0.25)
                    
                    if col_iqr > 0:  # Avoid division by zero
                        norm_col_name = f'{col}_robust_norm'
                        new_features_df[norm_col_name] = (col_filled - col_median) / col_iqr
                        behavior_features.append(norm_col_name)
                    else:
                        # Handle constant columns
                        new_features_df[f'{col}_robust_norm'] = 0.0

                if behavior_features:
                    # Enhanced engagement score with weighted features
                    weights = np.array([1.0] * len(behavior_features))  # Equal weights by default
                    # Adjust weights based on feature importance if available
                    if hasattr(data, 'feature_importance'):
                        weights = data.feature_importance[behavior_features]
                        weights = weights / weights.sum()  # Normalize weights
                    
                    new_features_df['engagement_score'] = (new_features_df[behavior_features] * weights).sum(axis=1)
                    print(f"* Created weighted engagement score from {len(behavior_features)} normalized features")
                else:
                    new_features_df['engagement_score'] = 0.0
                    print("* No valid engagement features found to create score.")

            except Exception as e:
                print(f"* Warning: Could not create engagement score: {str(e)}")
                if 'engagement_score' not in new_features_df: new_features_df['engagement_score'] = 0.0

        # Lead Quality Features with enhanced scoring
        quality_keywords = ['income', 'credit', 'budget', 'qualif', 'eligible', 'grade', 'tier', 'score', 'value', 'worth', 'equity']
        quality_cols = []
        for kw in quality_keywords:
             matching = [col for col in data.columns if kw.lower() in col.lower() and pd.api.types.is_numeric_dtype(data[col])]
             quality_cols.extend(matching)
        quality_cols = sorted(list(set([col for col in quality_cols if 'engagement' not in col.lower()])))
        print(f"* Identified {len(quality_cols)} potential numeric lead quality indicators: {quality_cols}")

        quality_features = []
        if quality_cols:
            try:
                for col in quality_cols:
                    # Enhanced z-score calculation with robust statistics
                    col_filled = data[col].fillna(data[col].median())
                    col_median = col_filled.median()
                    col_mad = np.median(np.abs(col_filled - col_median))  # Median Absolute Deviation
                    
                    if col_mad > 0:  # Avoid division by zero
                        zscore_col_name = f'{col}_robust_zscore'
                        new_features_df[zscore_col_name] = (col_filled - col_median) / col_mad
                        quality_features.append(zscore_col_name)
                    else:
                        new_features_df[f'{col}_robust_zscore'] = 0.0

                if quality_features:
                    # Enhanced quality score with weighted features
                    weights = np.array([1.0] * len(quality_features))  # Equal weights by default
                    # Adjust weights based on feature importance if available
                    if hasattr(data, 'feature_importance'):
                        weights = data.feature_importance[quality_features]
                        weights = weights / weights.sum()  # Normalize weights
                    
                    new_features_df['lead_quality_score'] = (new_features_df[quality_features] * weights).sum(axis=1)
                    print(f"* Created weighted lead quality score from {len(quality_features)} z-scored features")
                else:
                    new_features_df['lead_quality_score'] = 0.0
                    print("* No valid quality features found to create score.")

            except Exception as e:
                print(f"* Warning: Could not create lead quality score: {str(e)}")
                if 'lead_quality_score' not in new_features_df: new_features_df['lead_quality_score'] = 0.0

        # Enhanced Interaction Features
        print("* Creating enhanced interaction features...")
        interaction_created_count = 0
        try:
            # Collect all interaction features in dictionaries
            interaction_train = {}
            interaction_test = {}
            
            # Quality-Engagement Interactions
            if 'lead_quality_score' in new_features_df.columns and 'engagement_score' in new_features_df.columns:
                interaction_train['quality_engagement_interaction'] = new_features_df['lead_quality_score'] * new_features_df['engagement_score']
                interaction_train['quality_engagement_ratio'] = new_features_df['lead_quality_score'] / (new_features_df['engagement_score'] + 1e-6)
                print("  - Created quality-engagement interactions")
                interaction_created_count += 2

            # Time-Quality Interactions
            if best_date_col and 'lead_quality_score' in new_features_df.columns:
                for time_feature in ['is_morning', 'is_afternoon', 'is_evening', 'is_night', 'is_weekend']:
                    if f'{best_date_col}_{time_feature}' in new_features_df.columns:
                        interaction_train[f'{time_feature}_quality_interaction'] = new_features_df[f'{best_date_col}_{time_feature}'] * new_features_df['lead_quality_score']
                        interaction_created_count += 1

            # Season-Quality Interactions
            if best_date_col and 'lead_quality_score' in new_features_df.columns:
                for season in ['is_winter', 'is_spring', 'is_summer', 'is_fall']:
                    if f'{best_date_col}_{season}' in new_features_df.columns:
                        interaction_train[f'{season}_quality_interaction'] = new_features_df[f'{best_date_col}_{season}'] * new_features_df['lead_quality_score']
                        interaction_created_count += 1

            if interaction_created_count == 0:
                print("  - No interaction features could be created (required source features might be missing).")
            else:
                # Convert interaction features to DataFrame and concatenate
                interaction_df = pd.DataFrame(interaction_train, index=new_features_df.index)
                new_features_df = pd.concat([new_features_df, interaction_df], axis=1)

        except Exception as e:
             print(f"* Warning: Could not create some interaction features: {str(e)}")

        # Location Features (Simplified - using Zip prefix or State)
        loc_col = None
        loc_type = None
        # Prioritize zip/postal
        for col in data.columns:
            if ('zip' in col.lower() or 'postal' in col.lower()) and data[col].nunique() > 1: # Ensure it's not constant
                 # Check if it looks like a zip code (mostly digits, maybe hyphens)
                 if data[col].astype(str).str.match(r'^\d{3,}').mean() > 0.5:
                     loc_col = col
                     loc_type = 'zip'
                     break
        # Fallback to state if no zip found
        if not loc_col:
            for col in data.columns:
                if 'state' in col.lower() and data[col].nunique() > 1 and data[col].nunique() < 100: # Reasonable number of states/provinces
                     loc_col = col
                     loc_type = 'state'
                     break
        # Fallback to city if still no location found
        if not loc_col:
             for col in data.columns:
                  if 'city' in col.lower() and data[col].nunique() > 1 and data[col].nunique() < max_categories * 2: # Allow higher cardinality for city
                       loc_col = col
                       loc_type = 'city'
                       break

        if loc_col:
             print(f"* Using '{loc_col}' (type: {loc_type}) for location features.")
             try:
                  # Create region feature based on type
                  if loc_type == 'zip':
                       # Use first 3 digits as region, fill NaNs/errors with 'UNK'
                       new_features_df[f'{loc_col}_region'] = data[loc_col].astype(str).str.extract(r'^(\d{3})')[0].fillna('UNK')
                       print(f"  - Created region feature (first 3 digits) from '{loc_col}'")
                  elif loc_type == 'state':
                       # Use state directly, fill NaNs with 'UNK'
                       new_features_df[f'{loc_col}_region'] = data[loc_col].astype(str).fillna('UNK')
                       print(f"  - Using state directly as region feature from '{loc_col}'")
                  elif loc_type == 'city':
                       # Use city directly if cardinality is manageable, otherwise maybe use frequency encoding or drop later
                       if data[loc_col].nunique() <= max_categories:
                            new_features_df[f'{loc_col}_region'] = data[loc_col].astype(str).fillna('UNK')
                            print(f"  - Using city directly as region feature from '{loc_col}'")
                       else:
                            print(f"  - Cardinality for city '{loc_col}' ({data[loc_col].nunique()}) is high, skipping direct region feature.")
                            # Optional: Could add frequency encoding here
                            # city_map = data[loc_col].value_counts(normalize=True)
                            # new_features_df[f'{loc_col}_freq'] = data[loc_col].map(city_map).fillna(0)

             except Exception as e:
                  print(f"* Warning: Could not create region feature from {loc_col}: {str(e)}")
        else:
             print("* No suitable column found for location features (Zip/State/City).")

        # --- Concatenate new features to the original DataFrame ---
        # Get list of columns actually added
        new_feature_names = new_features_df.columns.tolist()
        # Drop any columns that might be all NaN (e.g., if a feature failed entirely)
        new_features_df = new_features_df.dropna(axis=1, how='all')
        final_new_feature_names = new_features_df.columns.tolist()
        dropped_nan_cols = list(set(new_feature_names) - set(final_new_feature_names))
        if dropped_nan_cols:
            print(f"* Dropping newly created features that were all NaN: {dropped_nan_cols}")

        new_features_added = len(final_new_feature_names)
        print(f"* Feature engineering complete - adding {new_features_added} new features...")

        # Ensure no duplicate columns before concat (e.g., if a generated name clashed)
        cols_to_concat = [col for col in final_new_feature_names if col not in data.columns]
        if len(cols_to_concat) < len(final_new_feature_names):
             print(f"* Warning: Some generated feature names already existed in original data. Only adding non-duplicates.")

        if cols_to_concat:
             data = pd.concat([data, new_features_df[cols_to_concat]], axis=1)

        print(f"* Final DataFrame shape after feature engineering: {data.shape}")
    else:
        print("* Skipping enhanced feature engineering.")


    # 5. Identify Feature Types (Re-run after potential feature additions)
    print("\n5. Identify Feature Types...")
    print("-"*40)
    categorical_cols = []
    numerical_cols = []
    high_cardinality_cols = []
    temp_date_cols_to_drop = [] # Track date columns added temporarily

    for col in data.columns:
        # Skip known temporary columns or ID columns
        if col == id_col_name or '_dt' in col: # Simple check for datetime objects added earlier
             if '_dt' in col: temp_date_cols_to_drop.append(col)
             continue

        # Convert potential boolean types represented as objects/strings to int
        if data[col].dtype == 'object' or str(data[col].dtype) == 'string':
            try:
                 # Attempt conversion to numeric, coerce errors to NaN
                 num_col = pd.to_numeric(data[col], errors='coerce')
                 # If conversion results in mostly numeric (few NaNs), treat as numeric
                 if num_col.notna().mean() > 0.95 and pd.api.types.is_numeric_dtype(num_col):
                      data[col] = num_col # Replace original column
                      # Fall through to numeric check below
                 # Else, check if it looks like a boolean after filling NaNs
                 elif data[col].fillna(-1).nunique(dropna=False) <= 3: # Check for 0/1, True/False, maybe NaN
                       # Try converting common bool representations
                       bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 1:1, 0:0}
                       # Apply map robustly, keep non-matches as NaN initially
                       maybe_bool_col = data[col].astype(str).str.lower().map(bool_map)
                       # If most values mapped successfully, treat as numeric (0/1)
                       if maybe_bool_col.notna().mean() > 0.9:
                           print(f"* Converting potential boolean column '{col}' to numeric (0/1)")
                           data[col] = maybe_bool_col.fillna(0) # Fill remaining NaNs with 0
                           numerical_cols.append(col)
                           continue # Skip further checks for this column
                       # Otherwise, proceed to categorical check
            except Exception:
                 pass # Ignore conversion errors, proceed as object/string

        n_unique = data[col].nunique()
        # Categorical: object/string OR low-cardinality integer that's not a float
        is_object_or_string = data[col].dtype == 'object' or str(data[col].dtype) == 'string'
        # Treat integer columns with low cardinality as categorical unless they are clearly flags (0/1)
        is_low_card_int = pd.api.types.is_integer_dtype(data[col]) and (n_unique > 2 and n_unique < 20)
        is_potential_cat_float = pd.api.types.is_float_dtype(data[col]) and n_unique < 10 and (data[col].apply(lambda x: x == int(x) if pd.notna(x) else True)).all() # Floats that are all integers

        if is_object_or_string or is_low_card_int or is_potential_cat_float:
            if n_unique <= max_categories:
                categorical_cols.append(col)
                # Ensure categorical columns are treated as strings for consistent encoding
                if not is_object_or_string: # Convert numeric categoricals to string
                     data[col] = data[col].astype(str)
            else:
                # Check if high-cardinality column is numeric-like string (e.g., IDs, long numbers)
                 if is_object_or_string and pd.to_numeric(data[col], errors='coerce').notna().mean() > 0.8:
                      print(f"* High-cardinality column '{col}' ({n_unique} values) looks numeric, treating as numerical.")
                      try:
                          data[col] = pd.to_numeric(data[col], errors='coerce') # Convert to numeric
                          numerical_cols.append(col)
                      except Exception as conv_err:
                           print(f"  - Failed converting '{col}' to numeric: {conv_err}. Marking for drop.")
                           high_cardinality_cols.append(col)
                 else:
                      high_cardinality_cols.append(col)
                      print(f"* Marking high-cardinality non-numeric column '{col}' ({n_unique} values) for potential drop")

        elif pd.api.types.is_numeric_dtype(data[col]): # Includes float, high-cardinality int, and bools (0/1)
            numerical_cols.append(col)
        else:
            # This case should ideally not be reached if types are standard
            print(f"* Skipping column '{col}' with unhandled dtype: {data[col].dtype}")

    # Drop high-cardinality non-numeric columns identified
    if high_cardinality_cols:
        data = data.drop(columns=high_cardinality_cols, errors='ignore')
        print(f"* Dropped {len(high_cardinality_cols)} high-cardinality non-numeric columns: {high_cardinality_cols}")

    # Drop temporary datetime columns if they exist
    if temp_date_cols_to_drop:
         data = data.drop(columns=[col for col in temp_date_cols_to_drop if col in data.columns], errors='ignore')
         print(f"* Dropped temporary datetime columns used for feature generation: {temp_date_cols_to_drop}")


    # Ensure no overlap between lists
    categorical_cols = sorted([col for col in categorical_cols if col in data.columns])
    numerical_cols = sorted([col for col in numerical_cols if col in data.columns and col not in categorical_cols])

    print(f"* Identified {len(categorical_cols)} categorical columns.")
    print(f"* Identified {len(numerical_cols)} numerical columns.")

    # 6. Handling Missing Values (Final Pass)
    print("\n6. Handling Missing Values...")
    print("-"*40)
    # Numerical: impute with median, handle infinity, clip extremes
    if numerical_cols:
        numerical_data = data[numerical_cols].copy() # Work on a copy

        cols_with_all_nan = numerical_data.columns[numerical_data.isnull().all()]
        if not cols_with_all_nan.empty:
            print(f"* Warning: {len(cols_with_all_nan)} numerical columns have all NaN values. Dropping: {cols_with_all_nan.tolist()}")
            numerical_data = numerical_data.drop(columns=cols_with_all_nan)
            numerical_cols = numerical_data.columns.tolist() # Update list
            if not numerical_cols: # Check if all numerical columns were dropped
                 print("* All numerical columns dropped due to being all NaN.")


        # Handle infinity first before imputation
        inf_replaced_cols = []
        if numerical_cols and numerical_data.shape[1] > 0: # Check if dataframe is not empty
             for col in numerical_cols:
                  # Check for inf/-inf efficiently
                  if np.isinf(numerical_data[col].values).any():
                       numerical_data[col] = numerical_data[col].replace([np.inf, -np.inf], np.nan)
                       inf_replaced_cols.append(col)
             if inf_replaced_cols: print(f"* Replaced Inf values with NaN in columns: {inf_replaced_cols}")

             # Impute with median (more robust to outliers than mean)
             print("* Imputing missing numerical values using column medians...")
             medians = numerical_data.median()
             numerical_data = numerical_data.fillna(medians)

             # If any NaNs remain (e.g., median was NaN for a column), fill with 0
             if numerical_data.isnull().any().any():
                 nan_cols_after_median = numerical_data.columns[numerical_data.isnull().any()].tolist()
                 print(f"* Filling remaining NaNs with 0 after median imputation in: {nan_cols_after_median}")
                 numerical_data = numerical_data.fillna(0)

             # Clip extreme values (using 1st and 99th percentiles)
             print("* Clipping extreme numerical values (outside ~1.5*IQR of 1st-99th percentile)...")
             clipped_cols_count = 0
             for col in numerical_cols:
                  q01 = numerical_data[col].quantile(0.01)
                  q99 = numerical_data[col].quantile(0.99)
                  # Ensure quantiles are valid and different before calculating bounds
                  if pd.notna(q01) and pd.notna(q99) and q99 > q01:
                       iqr_ext = (q99 - q01) * 1.5 # Extended IQR
                       lower_bound = q01 - iqr_ext
                       upper_bound = q99 + iqr_ext
                       initial_min, initial_max = numerical_data[col].min(), numerical_data[col].max()
                       # Clip the column
                       numerical_data[col] = numerical_data[col].clip(lower=lower_bound, upper=upper_bound)
                       # Check if clipping actually occurred
                       if numerical_data[col].min() > initial_min or numerical_data[col].max() < initial_max:
                            clipped_cols_count += 1
                  # else: print(f"  - Skipping clipping for column '{col}' (constant or invalid quantiles)") # Optional verbose logging
             if clipped_cols_count > 0: print(f"* Clipping applied to {clipped_cols_count} numerical columns.")

             # Update the original data DataFrame with processed numerical data
             data[numerical_cols] = numerical_data
        else:
             print("* No numerical columns to process for missing values or clipping.")


    # Categorical: impute with mode 'missing' string
    if categorical_cols:
        categorical_data = data[categorical_cols].copy() # Work on a copy
        imputed_cat_cols_count = 0
        print("* Imputing missing categorical values with string 'missing'...")
        for col in categorical_cols:
             # Ensure column is string type before checking for NaNs/filling
             categorical_data[col] = categorical_data[col].astype(str)
             if categorical_data[col].isnull().any() or (categorical_data[col] == 'nan').any(): # Check for actual None/NaN and string 'nan'
                  fill_value = 'missing' # Explicit string marker
                  # Replace both actual NaNs and string 'nan'
                  categorical_data[col] = categorical_data[col].fillna(fill_value)
                  categorical_data[col] = categorical_data[col].replace('nan', fill_value)
                  imputed_cat_cols_count += 1

        if imputed_cat_cols_count > 0: print(f"* Imputation applied to {imputed_cat_cols_count} categorical columns.")
        # Update the original data DataFrame
        data[categorical_cols] = categorical_data
    else:
        print("* No categorical columns to process for missing values.")


    # 7. Train/Test Split
    print("\n7. Train/Test Split...")
    print("-"*40)
    # Ensure indices are aligned before splitting
    data = data.reset_index(drop=True)
    y_toured = y_toured.reset_index(drop=True)
    lead_ids = lead_ids.reset_index(drop=True)

    # Select only the final feature columns for splitting
    final_feature_cols = categorical_cols + numerical_cols
    if not final_feature_cols:
         raise ValueError("No features remaining after preprocessing!")
    X = data[final_feature_cols]

    # Stratify based on the toured target if possible
    stratify_target = y_toured if y_toured.nunique() > 1 else None
    if stratify_target is None:
        print("* Warning: Cannot stratify split - target 'toured' has only one class.")

    # Split indices first to avoid copying large dataframes
    indices = np.arange(len(X))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target.iloc[indices] if stratify_target is not None else None # Use .iloc for Series stratification
        )
    except ValueError as split_err:
         print(f"Warning: Stratified split failed ({split_err}). Performing non-stratified split.")
         train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None
         )


    # Create split dataframes using indices
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train_toured = y_toured.iloc[train_indices]
    y_test_toured = y_toured.iloc[test_indices]
    train_ids = lead_ids.iloc[train_indices]
    test_ids = lead_ids.iloc[test_indices]

    print(f"* Training set size: {len(X_train):,}")
    print(f"* Testing set size:  {len(X_test):,}")
    print(f"* Train target distribution: {y_train_toured.mean()*100:.2f}% positive")
    print(f"* Test target distribution:  {y_test_toured.mean()*100:.2f}% positive")

    # 8. Preprocessing (Encoding & Scaling) on Splits
    print("\n8. Preprocessing (Encoding & Scaling)...")
    print("-"*40)

    # Categorical Encoding (Label Encoding)
    categorical_encoders = {}
    X_train_cat_encoded = pd.DataFrame(index=X_train.index)
    X_test_cat_encoded = pd.DataFrame(index=X_test.index)
    categorical_dims = [] # Stores number of unique categories for each feature

    if categorical_cols:
        print("* Applying Label Encoding to categorical features...")
        # Use tqdm for progress visualization
        encoded_train_cols = {}
        encoded_test_cols = {}
        
        for col in tqdm(categorical_cols, desc="Encoding Categories"):
            if col not in X_train.columns: # Should not happen with previous checks, but safeguard
                 print(f"  - Warning: Categorical column '{col}' not found in split data. Skipping.")
                 continue
            encoder = LabelEncoder()
            # Combine train and test (as strings) to fit the encoder on all possible values seen in the data
            # Ensure the column is string type before combining/fitting
            combined_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)

            # Fit encoder
            encoder.fit(combined_values)

            # Transform train and test (ensure string type for transform too)
            encoded_train_cols[col] = encoder.transform(X_train[col].astype(str))
            encoded_test_cols[col] = encoder.transform(X_test[col].astype(str))

            # Store encoder and dimensions (number of unique classes found)
            categorical_encoders[col] = encoder
            categorical_dims.append(len(encoder.classes_))
            
        # Convert encoded columns to DataFrames
        X_train_cat_encoded = pd.DataFrame(encoded_train_cols, index=X_train.index)
        X_test_cat_encoded = pd.DataFrame(encoded_test_cols, index=X_test.index)
        
        print(f"* Encoded {len(categorical_cols)} categorical features.")
        if categorical_dims: print(f"* Categorical dimensions (for embedding layers): {dict(zip(categorical_cols, categorical_dims))}")
    else:
        print("* No categorical features to encode.")
        # Ensure tensors are created with correct empty shape later


    # Numerical Scaling (Standard Scaler)
    scaler = None
    X_train_num_scaled = np.zeros((len(X_train), 0)) # Default empty array for tensor creation
    X_test_num_scaled = np.zeros((len(X_test), 0))   # Default empty array

    if numerical_cols:
        print("* Applying Standard Scaling to numerical features...")
        # Select numerical columns safely from the split data
        X_train_num = X_train[[col for col in numerical_cols if col in X_train.columns]]
        X_test_num = X_test[[col for col in numerical_cols if col in X_test.columns]]

        if not X_train_num.empty:
             scaler = StandardScaler()
             try:
                 # Fit only on training data
                 X_train_num_scaled = scaler.fit_transform(X_train_num)
                 # Transform test data
                 X_test_num_scaled = scaler.transform(X_test_num)

                 # Final check for NaNs after scaling (shouldn't happen with prior imputation/handling)
                 if np.isnan(X_train_num_scaled).any() or np.isnan(X_test_num_scaled).any():
                      print("  * Warning: NaNs detected after scaling! Check imputation/clipping. Filling NaNs with 0.")
                      X_train_num_scaled = np.nan_to_num(X_train_num_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                      X_test_num_scaled = np.nan_to_num(X_test_num_scaled, nan=0.0, posinf=0.0, neginf=0.0)

                 print(f"* Scaled {X_train_num_scaled.shape[1]} numerical features.")
             except Exception as scale_err:
                  print(f"* Error during scaling: {scale_err}. Numerical features will not be scaled.")
                  # Reset scaled arrays to original (or keep as empty if preferred)
                  X_train_num_scaled = X_train_num.values # Use original values if scaling fails
                  X_test_num_scaled = X_test_num.values
                  scaler = None # Indicate scaling failed/was skipped
        else:
            print("* No numerical features found in the split data to scale.")
    else:
        print("* No numerical features identified to scale.")

    numerical_dim = X_train_num_scaled.shape[1]


    # 9. Create Datasets
    print("\n9. Creating PyTorch Datasets...")
    print("-"*40)
    # Convert to tensors
    try:
        # Handle cases where there are no categorical or numerical features
        if not X_train_cat_encoded.empty:
            X_train_cat_tensor = torch.tensor(X_train_cat_encoded.values, dtype=torch.long)
            X_test_cat_tensor = torch.tensor(X_test_cat_encoded.values, dtype=torch.long)
        else:
            X_train_cat_tensor = torch.zeros((len(X_train), 0), dtype=torch.long)
            X_test_cat_tensor = torch.zeros((len(X_test), 0), dtype=torch.long)

        if X_train_num_scaled.shape[1] > 0:
            X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
            X_test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32)
        else:
            X_train_num_tensor = torch.zeros((len(X_train), 0), dtype=torch.float32)
            X_test_num_tensor = torch.zeros((len(X_test), 0), dtype=torch.float32)


        y_train_toured_tensor = torch.tensor(y_train_toured.values, dtype=torch.float32).view(-1, 1)
        y_test_toured_tensor = torch.tensor(y_test_toured.values, dtype=torch.float32).view(-1, 1)

        # Convert IDs safely (handle potential non-numeric IDs if necessary)
        try:
             # Attempt direct conversion first
             train_ids_tensor = torch.tensor(train_ids.values)
             test_ids_tensor = torch.tensor(test_ids.values)
             # Check if tensor conversion resulted in float (unexpected for IDs)
             if train_ids_tensor.dtype == torch.float32 or train_ids_tensor.dtype == torch.float64:
                 print("Warning: Lead IDs converted to float tensor, attempting long conversion.")
                 train_ids_tensor = train_ids_tensor.long()
                 test_ids_tensor = test_ids_tensor.long()

        except (TypeError, ValueError):
             print("Warning: Lead IDs could not be directly converted to numeric tensor. Using index as fallback ID.")
             # Fallback to using DataFrame index as ID
             train_ids_tensor = torch.tensor(train_ids.index.values, dtype=torch.long)
             test_ids_tensor = torch.tensor(test_ids.index.values, dtype=torch.long)

        # Create dataset objects using the imported/defined LeadDataset
        # Pass only y_toured; the placeholder LeadDataset creates dummy tensors for others
        train_dataset = LeadDataset(
            X_train_cat_tensor, X_train_num_tensor,
            y_train_toured_tensor,
            lead_ids=train_ids_tensor
        )
        test_dataset = LeadDataset(
            X_test_cat_tensor, X_test_num_tensor,
            y_test_toured_tensor,
            lead_ids=test_ids_tensor
        )
        print(f"* Created Train Dataset with {len(train_dataset)} samples.")
        print(f"* Created Test Dataset with {len(test_dataset)} samples.")
        print(f"* Sample Train Batch structure: Cat({X_train_cat_tensor.shape}), Num({X_train_num_tensor.shape}), Toured({y_train_toured_tensor.shape}), Applied(dummy), Rented(dummy), ID({train_ids_tensor.shape})")


    except Exception as e:
        print(f"Error creating tensors or datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


    # 10. Save Preprocessors
    if save_preprocessors:
        print("\n10. Saving Preprocessors...")
        print("-"*40)
        try:
            if not os.path.exists(preprocessors_path):
                os.makedirs(preprocessors_path)
                print(f"* Created preprocessor directory: {preprocessors_path}")

            # Combine all necessary objects for prediction/inference later
            preprocessors = {
                'categorical_encoders': categorical_encoders, # Dict of {col_name: LabelEncoder}
                'scaler': scaler, # StandardScaler object (or None if scaling failed/skipped)
                'categorical_cols': categorical_cols, # List of categorical column names in order
                'numerical_cols': numerical_cols, # List of numerical column names in order
                'feature_names': categorical_cols + numerical_cols, # Combined list in processing order
                # Store dims needed for model architecture definition
                'categorical_dims': categorical_dims, # List of dimensions/vocab sizes for embeddings
                'numerical_dim': numerical_dim, # Number of numerical features
                # Optional: Store other metadata for reference
                'target_col': target_col,
                'random_state': random_state,
                'max_categories': max_categories,
                'enhance_features': enhance_features
            }
            preprocessor_file_path = os.path.join(preprocessors_path, 'tour_preprocessors.joblib')

            joblib.dump(preprocessors, preprocessor_file_path)
            print(f"* Saved preprocessors to {preprocessor_file_path}")
        except Exception as e:
            print(f"Error saving preprocessors: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without saving if it fails, but warn user


    print("\n" + "="*80)
    print("TOUR DATA PREPARATION COMPLETE")
    print("="*80 + "\n")

    # Return datasets and model configuration info needed for training
    return (
        train_dataset,
        test_dataset,
        categorical_dims, # List of vocab sizes for embeddings
        numerical_dim,    # Number of numerical features
        categorical_cols + numerical_cols # Final list of feature names used
    )

# Example usage block for testing
if __name__ == '__main__':
    # Define arguments similar to how they might be passed from a script
    # --- IMPORTANT: UPDATE data_path TO YOUR ACTUAL FILE ---
    test_data_path = 'path/to/your/data.csv' # <--- CHANGE THIS
    # ---

    class Args:
        data_path = test_data_path
        dict_path = None # Optional: path/to/dict.csv
        dict_map_path = None # Optional: path/to/map.csv
        target_toured = 'TOTAL_APPOINTMENT_COMPLETED' # Or your specific tour target column
        enhance_features = True # Set to False to skip detailed feature engineering
        seed = 123
        preprocessor_dir = './preprocessors/tour_test_run' # Save to a test directory

    args = Args()
    print("--- Running Example Usage ---")

    if not os.path.exists(args.data_path):
        print(f"\n!!! ERROR: Data file not found at '{args.data_path}'")
        print("!!! Please update the 'test_data_path' variable in the if __name__ == '__main__' block.\n")
    else:
        print(f"Using data file: {args.data_path}")
        print(f"Enhanced features: {args.enhance_features}")
        print(f"Target column: {args.target_toured}")
        print(f"Saving preprocessors to: {args.preprocessor_dir}")

        try:
            train_ds, test_ds, cat_dims, num_dim, feat_names = prepare_tour_data(
                data_path=args.data_path,
                dict_path=args.dict_path,
                dict_map_path=args.dict_map_path,
                target_col=args.target_toured,
                enhance_features=args.enhance_features,
                random_state=args.seed,
                save_preprocessors=True,
                preprocessors_path=args.preprocessor_dir
            )
            print("\n--- Example Usage Results ---")
            print(f"Successfully prepared data.")
            print(f"Train dataset size: {len(train_ds)}")
            print(f"Test dataset size: {len(test_ds)}")
            print(f"Number of categorical features: {len(cat_dims)}")
            # print(f"Categorical dimensions (first 5): {cat_dims[:5]}")
            print(f"Numerical dimension: {num_dim}")
            print(f"Total features processed: {len(feat_names)}")
            # print(f"Feature names (first 10): {feat_names[:10]}")

            # You can inspect a sample batch here if needed
            # from torch.utils.data import DataLoader
            # train_loader = DataLoader(train_ds, batch_size=4)
            # sample_batch = next(iter(train_loader))
            # print("\nSample batch structure:")
            # print(f"  Categorical: {sample_batch[0].shape}, dtype={sample_batch[0].dtype}")
            # print(f"  Numerical: {sample_batch[1].shape}, dtype={sample_batch[1].dtype}")
            # print(f"  Toured Target: {sample_batch[2].shape}, dtype={sample_batch[2].dtype}")
            # print(f"  Applied Target (Dummy): {sample_batch[3].shape}, dtype={sample_batch[3].dtype}")
            # print(f"  Rented Target (Dummy): {sample_batch[4].shape}, dtype={sample_batch[4].dtype}")
            # print(f"  Lead IDs: {sample_batch[5].shape}, dtype={sample_batch[5].dtype}")


        except FileNotFoundError:
             print(f"\n!!! ERROR DURING EXECUTION: Data file not found at '{args.data_path}'")
             print("!!! Please ensure the path is correct.")
        except ValueError as ve:
             print(f"\n!!! ERROR DURING EXECUTION: {ve}")
             print("!!! Check if the target column exists or if feature processing failed.")
             import traceback
             traceback.print_exc()
        except Exception as e:
            print(f"\n!!! An unexpected error occurred during data preparation: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- End of Example Usage ---")
