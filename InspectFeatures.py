import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import joblib

class LeadDataset(Dataset):
    def __init__(self, categorical_features, numerical_features,
                 toured_labels=None, applied_labels=None, rented_labels=None,
                 lead_ids=None):
        self.categorical_features = categorical_features.long() if categorical_features is not None else None
        self.numerical_features = numerical_features.float() if numerical_features is not None else None
        self.toured_labels = toured_labels
        self.applied_labels = applied_labels
        self.rented_labels = rented_labels
        self.lead_ids = lead_ids

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        items = [
            self.categorical_features[idx] if self.categorical_features is not None else torch.tensor([], dtype=torch.long),
            self.numerical_features[idx]
        ]
        if self.toured_labels is not None:
            items.append(self.toured_labels[idx])
        if self.applied_labels is not None:
            items.append(self.applied_labels[idx])
        if self.rented_labels is not None:
            items.append(self.rented_labels[idx])
        if self.lead_ids is not None:
            items.append(self.lead_ids[idx])
        return tuple(items)

def preprocess_data(
    data_path,
    dict_path=None,  # Add dictionary args
    dict_map_path=None,
    target_cols=['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED'],
    test_size=0.2,
    random_state=42,
    max_categories=100,
    save_preprocessors=True,
    preprocessors_path='./preprocessors',
    enhance_toured_features=True,
    use_percentages=True,
    adapt_to_data=True
):
    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)

    print("\n1. Loading Data...")
    print("-"*40)
    data = pd.read_csv(data_path, low_memory=False)
    print(f"* Loaded data with {len(data):,} rows and {len(data.columns):,} columns")

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

    print("\n3. Creating Target Variables...")
    print("-"*40)
    y_toured = (data[target_cols[0]] > 0).astype(int)
    y_applied = (data[target_cols[1]] > 0).astype(int)
    y_rent = (data[target_cols[2]] > 0).astype(int)
    print(f"* TOURED:  {y_toured.mean()*100:.2f}% positive ({y_toured.sum():,} of {len(y_toured):,})")
    print(f"* APPLIED: {y_applied.mean()*100:.2f}% positive ({y_applied.sum():,} of {len(y_applied):,})")
    print(f"* RENTED:  {y_rent.mean()*100:.2f}% positive ({y_rent.sum():,} of {len(y_rent):,})")

    stage_rates = {
        'toured_rate': y_toured.mean(),
        'applied_rate': y_applied.mean(),
        'rented_rate': y_rent.mean(),
        'applied_given_toured': y_applied[y_toured == 1].mean(),
        'rented_given_applied': y_rent[y_applied == 1].mean()
    }
    print("\nStage Progression Rates:")
    for k, v in stage_rates.items():
        print(f"* {k}: {v*100:.2f}%")

    if adapt_to_data and use_percentages:
        toured_pct = min(1.0, stage_rates['toured_rate'] * 1.2)
        applied_pct = min(1.0, stage_rates['applied_given_toured'] * 1.2)
        rented_pct = min(1.0, stage_rates['rented_given_applied'] * 1.2)
        print(f"\nAdapted selection percentages:")
        print(f"* Toured: {toured_pct*100:.2f}%")
        print(f"* Applied: {applied_pct*100:.2f}%")
        print(f"* Rented: {rented_pct*100:.2f}%")

    print("\n4. Feature Engineering...")
    print("-"*40)
    original_cols = set(data.columns)
    print(f"Dropping target columns: {target_cols}")
    data = data.drop(columns=target_cols, errors='ignore')

    cols_to_drop = [
        "HASH", "RECD_LUID", "RECD_P_ID", "RECD_A_ID", "CLIENT_PERSON_ID", "CLIENT_ID", "RN",
        "FNAM_FNAM", "MNAM_MNAM", "PFXT_PFXT", "SNAM_SNAM", "SFXT_SFXT", "FIRST_NAME", "LAST_NAME", "EMAIL", "PHONE",
        "ADDRESS_LINE_1", "ADDRESS_LINE_2", "CITY", "STRT_NAME_I1", "STRT_POST_I1", "STRT_PRED_I1", "STRT_SUFX_I1",
        "QUALIFIED", "EXTRACT_DATE", "NCOA_MOVE_UPDATE_DATE", "NCOA_MOVE_UPDATE_METHOD_CODE",
        "EST_CURRENT_MORTGAGE_AMOUNT", "ENRICHMENTESTIMATED_CURRENT_MORTGAGE_AMT", "EST_MONTHLY_MORTGAGE_PAYMENT",
        "EST_AVAILABLE_EQUITY_LL", "ESTIMATED_AVAILABLE_EQUITY", "EQTY_LNDR_I1", "MORT_LNDR_I1", "REFI_LNDR_I1",
        "GROUP_ID", "GROUP_NAME", "LEAD_CREATED_AT", "CITY_PLAC_I1", "COUNTY_CODE_I1", "STATE_ABBR_I1", "STATE_I1",
        "RECD_ZIPC_I1", "SCDY_NUMB_I1", "SCDY_DESG_I1", "INVS_TYPE_I1", "PRCH_TYPE_I1", "TOTAL_WALK_IN"
    ]
    print(f"Dropping additional columns: {[col for col in cols_to_drop if col in data.columns]}")
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')

    if enhance_toured_features:
        print("\n4.1 Creating specialized features for toured prediction...")
        initial_cols = len(data.columns)

        date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
        print(f"* Identified {len(date_columns)} potential date/time columns")
        best_date_col = None
        best_corr = 0
        dt_cols = []
        for col in date_columns:
            try:
                data[f'{col}_dt'] = pd.to_datetime(data[col], errors='coerce')
                dt_cols.append(f'{col}_dt')
                data[f'{col}_dow'] = data[f'{col}_dt'].dt.dayofweek
                corr = abs(data[f'{col}_dow'].corr(y_toured, method='spearman'))
                if corr > best_corr and not pd.isna(corr):
                    best_corr = corr
                    best_date_col = col
            except:
                continue
        if best_date_col:
            print(f"* Using '{best_date_col}' for time-based features (corr={best_corr:.4f})")
            col = best_date_col
            data[f'{col}_month'] = data[f'{col}_dt'].dt.month
            data[f'{col}_day'] = data[f'{col}_dt'].dt.day
            data[f'{col}_hour'] = data[f'{col}_dt'].dt.hour
            data['time_category'] = pd.cut(data[f'{col}_hour'], bins=[-1, 5, 11, 17, 23], labels=[1, 2, 3, 4])

        # Use dictionary if available
        important_cols = []
        if data_dict is not None:
            lead_keywords = ['lead', 'customer', 'client', 'property', 'apartment', 'rent', 'tour', 'visit']
            for keyword in lead_keywords:
                matching_fields = data_dict[data_dict['LONG_DESCRIPTION'].str.contains(keyword, case=False, na=False)]
                important_cols.extend(matching_fields['FIELD_NAME'].tolist())
            print(f"* Identified {len(important_cols)} potentially important columns from data dictionary")

        quality_keywords = ['income', 'credit', 'budget', 'qual', 'eligib', 'grade', 'tier', 'score']
        quality_cols = [col for col in data.columns for kw in quality_keywords if kw in col.lower()]
        print(f"* Identified {len(quality_cols)} potential lead quality indicators")
        quality_features = []
        for col in quality_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()
                quality_features.append(f'{col}_zscore')
        if quality_features:
            print(f"* Creating lead_quality_score from {len(quality_features)} features")
            data['lead_quality_score'] = data[quality_features].mean(axis=1).fillna(data[quality_features].mean(axis=1).median())

        location_cols = [col for col in data.columns if any(kw in col.lower() for kw in ['zip', 'state', 'city', 'postal', 'region', 'area', 'location', 'address'])]
        print(f"* Identified {len(location_cols)} potential location columns")
        best_loc_col = max(location_cols, key=lambda x: data[x].nunique()) if location_cols else None
        if best_loc_col:
            print(f"* Using '{best_loc_col}' for location features ({data[best_loc_col].nunique()} unique values)")

        if 'time_category' in data.columns and f'{best_date_col}_dow' in data.columns:
            data['day_time_interaction'] = data[f'{best_date_col}_dow'] * 10 + data['time_category'].astype(float)

        data = data.drop(columns=dt_cols, errors='ignore')
        new_cols = [col for col in data.columns if col not in original_cols]
        print(f"* Feature engineering complete - added {len(new_cols)} new features")
        print("New columns:", new_cols)

    print("\n5. Feature Selection...")
    print("-"*40)
    X = data
    missing_pct = X.isnull().mean()
    cols_to_drop_missing = missing_pct[missing_pct > 0.95].index
    X = X.drop(columns=cols_to_drop_missing)
    print(f"* Dropped {len(cols_to_drop_missing)} columns with >95% missing values")

    lead_ids = pd.Series(np.arange(len(X))) if 'CLIENT_PERSON_ID' not in X.columns else X['CLIENT_PERSON_ID']
    print("* No CLIENT_PERSON_ID found, creating sequential IDs")

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 20]
    numerical_cols = [col for col in X.columns if col not in categorical_cols and not col.endswith('_dt')]
    print(f"* Identified {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")

    print("\n6. Handling Missing Values...")
    print("-"*40)
    numerical_data = X[numerical_cols].copy()
    for col in numerical_cols:
        if numerical_data[col].dtype in ['float64', 'int64']:
            outliers = ((numerical_data[col] < numerical_data[col].quantile(0.01)) |
                        (numerical_data[col] > numerical_data[col].quantile(0.99))).sum()
            if outliers > 0:
                print(f"* Clipping {outliers} extreme values in '{col}'")
                numerical_data[col] = numerical_data[col].clip(numerical_data[col].quantile(0.01), numerical_data[col].quantile(0.99))
            numerical_data[col] = numerical_data[col].fillna(numerical_data[col].median())
    categorical_data = X[categorical_cols].copy()
    for col in categorical_cols:
        categorical_data[col] = categorical_data[col].fillna(categorical_data[col].mode()[0] if not categorical_data[col].mode().empty else 'missing')

    print("\n7. Train/Test Split...")
    print("-"*40)
    X_train_cat, X_test_cat, X_train_num, X_test_num, \
    y_train_toured, y_test_toured, y_train_applied, y_test_applied, \
    y_train_rent, y_test_rent, train_ids, test_ids = train_test_split(
        categorical_data, numerical_data, y_toured, y_applied, y_rent, lead_ids,
        test_size=test_size, random_state=random_state, stratify=y_rent
    )

    X_train_cat_encoded = X_train_cat.copy()
    X_test_cat_encoded = X_test_cat.copy()
    categorical_encoders = {}
    cat_dims = []
    for col in X_train_cat.columns:
        le = LabelEncoder()
        combined_values = pd.concat([X_train_cat[col], X_test_cat[col]]).astype(str)
        le.fit(combined_values)
        X_train_cat_encoded[col] = le.transform(X_train_cat[col].astype(str))
        X_test_cat_encoded[col] = le.transform(X_test_cat[col].astype(str))
        categorical_encoders[col] = le
        cat_dims.append(len(le.classes_))

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_cat_tensor = torch.tensor(X_train_cat_encoded.values, dtype=torch.long)
    X_test_cat_tensor = torch.tensor(X_test_cat_encoded.values, dtype=torch.long)
    X_train_num_tensor = torch.tensor(X_train_num_scaled, dtype=torch.float32)
    X_test_num_tensor = torch.tensor(X_test_num_scaled, dtype=torch.float32)
    y_train_toured_tensor = torch.tensor(y_train_toured.values, dtype=torch.float32).view(-1, 1)
    y_test_toured_tensor = torch.tensor(y_test_toured.values, dtype=torch.float32).view(-1, 1)
    y_train_applied_tensor = torch.tensor(y_train_applied.values, dtype=torch.float32).view(-1, 1)
    y_test_applied_tensor = torch.tensor(y_test_applied.values, dtype=torch.float32).view(-1, 1)
    y_train_rent_tensor = torch.tensor(y_train_rent.values, dtype=torch.float32).view(-1, 1)
    y_test_rent_tensor = torch.tensor(y_test_rent.values, dtype=torch.float32).view(-1, 1)
    train_ids_tensor = torch.tensor(train_ids.values)
    test_ids_tensor = torch.tensor(test_ids.values)

    train_dataset = LeadDataset(X_train_cat_tensor, X_train_num_tensor,
                                y_train_toured_tensor, y_train_applied_tensor, y_train_rent_tensor,
                                train_ids_tensor)
    test_dataset = LeadDataset(X_test_cat_tensor, X_test_num_tensor,
                               y_test_toured_tensor, y_test_applied_tensor, y_test_rent_tensor,
                               test_ids_tensor)

    print(f"Created datasets - Train: {len(train_dataset)} examples, Test: {len(test_dataset)} examples")
    all_engineered_cols = new_cols  # Capture all engineered features before filtering
    feat_names = list(X_train_cat_encoded.columns) + list(X_train_num.columns)
    engineered_features = [col for col in all_engineered_cols if col in feat_names]  # Ensure all are kept

    if save_preprocessors:
        if not os.path.exists(preprocessors_path):
            os.makedirs(preprocessors_path)
        preprocessors = {
            'categorical_encoders': categorical_encoders,
            'scaler': scaler,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'feature_names': feat_names,
            'stage_rates': stage_rates,
            'engineered_features': engineered_features
        }
        joblib.dump(preprocessors, os.path.join(preprocessors_path, 'preprocessors.joblib'))
        print(f"Saved preprocessors to {os.path.join(preprocessors_path, 'preprocessors.joblib')}")

    return train_dataset, test_dataset, cat_dims, X_train_num_scaled.shape[1], feat_names, stage_rates

# Main execution
data = pd.read_csv('person_hh_number_for_ml.csv', low_memory=False)
original_cols = set(data.columns) - set(['TOTAL_APPOINTMENT_COMPLETED', 'TOTAL_APPLIED', 'TOTAL_RENTED'])

train_dataset, _, _, _, feat_names, _ = preprocess_data(
    data_path='person_hh_number_for_ml.csv',
    dict_path='data_dictonary.csv',
    dict_map_path='data_dictionary_mapping.csv',
    enhance_toured_features=True,
    use_percentages=True,
    adapt_to_data=True,
    random_state=42
)

engineered_features = [feat for feat in feat_names if feat not in original_cols]
print(f"Total features: {len(feat_names)}")
print(f"Original features: {len(original_cols)}")
print(f"Engineered features: {len(engineered_features)}")
print("Engineered features:", engineered_features)

cat_data = train_dataset.categorical_features.numpy()
num_data = train_dataset.numerical_features.numpy()
all_data = np.hstack([cat_data, num_data])
df_all = pd.DataFrame(all_data, columns=feat_names)
df_all['toured'] = train_dataset.toured_labels.numpy().flatten()
df_all['applied'] = train_dataset.applied_labels.numpy().flatten()
df_all['rented'] = train_dataset.rented_labels.numpy().flatten()
df_engineered = df_all[engineered_features + ['toured', 'applied', 'rented']]
df_engineered.to_csv('engineered_features.csv', index=False)
print("Saved to 'engineered_features.csv'")

corrs = df_engineered.corr(method='spearman')
for target in ['toured', 'applied', 'rented']:
    target_corrs = corrs[target].drop(['toured', 'applied', 'rented'])
    print(f"\nHigh correlations with {target}:")
    print(target_corrs[abs(target_corrs) > 0.3].sort_values(ascending=False))