"""
preprocess.py — Data Cleaning & Feature Engineering
=====================================================
WHAT THIS FILE DOES:
  Loads the raw IEEE-CIS fraud dataset, cleans it, engineers new features,
  handles class imbalance, and splits it into train/val/test sets.

WHY EACH STEP MATTERS:
  - Fraud datasets are heavily imbalanced (only 3.5% fraud). Without fixing
    this, the model just predicts "not fraud" every time and gets 96% accuracy
    while catching ZERO frauds — useless. We use SMOTE to fix this.
  - Missing values crash LightGBM if not handled. We fill numeric cols with
    median (robust to outliers) and categorical with "UNKNOWN".
  - Label encoding converts string categories (e.g. "Visa", "Mastercard")
    into numbers (0, 1, 2...) that the model can process.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─── Column configuration ─────────────────────────────────────────────────────
# These are the most important features from the IEEE-CIS dataset.
# We use a subset to keep memory usage under 4GB on 8GB RAM systems.
SELECTED_FEATURES = [
    'TransactionAmt',       # Amount of money transferred
    'ProductCD',            # Product code (W, H, C, S, R)
    'card1', 'card2', 'card4', 'card6',  # Card type info
    'addr1', 'addr2',       # Billing address zip codes
    'dist1',                # Distance between buyer and seller
    'P_emaildomain',        # Purchaser email domain (gmail, yahoo, etc.)
    'R_emaildomain',        # Recipient email domain
    'C1', 'C2', 'C6', 'C13', 'C14',  # Count features (how many transactions)
    'D1', 'D10', 'D15',     # Timedelta features (days since last transaction)
    'M4', 'M5', 'M6',       # Match features (name/address match flags)
    'V12', 'V13', 'V29', 'V30', 'V33', 'V34',  # Vesta engineered features
    'TransactionDT',        # Transaction timestamp (seconds offset)
]

TARGET_COL = 'isFraud'

# These columns contain text/categories that need label encoding
CATEGORICAL_COLS = ['ProductCD', 'card4', 'card6', 'P_emaildomain',
                    'R_emaildomain', 'M4', 'M5', 'M6']


def load_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load and merge the two IEEE-CIS CSV files.
    
    The dataset comes in two files:
      - train_transaction.csv : the transaction details (amount, card, etc.)
      - train_identity.csv    : device/browser info (optional, for advanced use)
    
    We merge them on 'TransactionID'. Identity file adds device info but is
    optional — we skip it if not present to keep memory usage low.
    """
    print("📂 Loading dataset...")
    
    trans_path = os.path.join(data_dir, 'train_transaction.csv')
    
    if not os.path.exists(trans_path):
        raise FileNotFoundError(
            f"Dataset not found at {trans_path}\n"
            "Please download from: https://www.kaggle.com/c/ieee-fraud-detection\n"
            "and place CSV files inside the 'data/' folder."
        )
    
    # Load only the columns we need — saves ~2GB of RAM
    available_cols = SELECTED_FEATURES + [TARGET_COL]
    df = pd.read_csv(trans_path, usecols=lambda c: c in available_cols)
    
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Fraud rate: {df[TARGET_COL].mean():.2%}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that help the model spot fraud patterns.
    
    WHY NEW FEATURES HELP:
      Raw transaction data doesn't tell us "this amount is unusually large
      for this card". By computing ratios and time-based features, we give
      the model richer signals.
    """
    df = df.copy()
    
    # 1. Extract hour of day from TransactionDT
    #    Fraud often happens at unusual hours (2-5 AM)
    #    TransactionDT is seconds since a reference time — modulo gives hour
    if 'TransactionDT' in df.columns:
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
        df.drop('TransactionDT', axis=1, inplace=True)
    
    # 2. Log of transaction amount
    #    Amounts range from $0.25 to $31,937 — log scale helps the model
    #    compare a $100 and $200 transaction more naturally than $100 vs $200
    if 'TransactionAmt' in df.columns:
        df['log_amount'] = np.log1p(df['TransactionAmt'])
    
    # 3. Flag unusually large amounts
    #    Transactions > $1000 are rare and more likely to be fraud
    if 'TransactionAmt' in df.columns:
        df['is_large_amount'] = (df['TransactionAmt'] > 1000).astype(int)
    
    print("✅ Feature engineering complete")
    print(f"   Total features: {df.shape[1] - 1}")  # -1 for target
    
    return df


def clean_and_encode(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Handle missing values and encode categorical columns.
    
    PARAMETERS:
      df       : the dataframe to clean
      encoders : dict of pre-fitted LabelEncoders (for test/inference time)
      fit      : True = fit new encoders (training), False = use existing ones
    
    RETURNS:
      (cleaned dataframe, encoders dict)
    
    WHY SEPARATE fit=True and fit=False:
      At training time, we learn what "Visa" maps to (e.g. label 3).
      At inference time (new transaction comes in), we MUST use the same
      mapping — otherwise "Visa" might get mapped to a different number.
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    # Fill missing values
    # Numeric columns → median (more robust than mean for skewed distributions)
    # Categorical → 'UNKNOWN' string
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            fill_val = df[col].median() if fit else encoders.get(f'median_{col}', 0)
            df[col] = df[col].fillna(fill_val)
            if fit:
                encoders[f'median_{col}'] = fill_val
    
    # Encode categorical columns
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna('UNKNOWN').astype(str)
        
        if fit:
            le = LabelEncoder()
            # Add 'UNKNOWN' to handle unseen categories at inference
            all_values = list(df[col].unique()) + ['UNKNOWN']
            le.fit(all_values)
            encoders[col] = le
        
        le = encoders[col]
        # Handle values not seen during training
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else 'UNKNOWN')
        df[col] = le.transform(df[col])
    
    print("✅ Missing values filled, categories encoded")
    return df, encoders


def split_and_balance(df: pd.DataFrame):
    """
    Split data into train/val/test and apply SMOTE to training set.
    
    SPLIT RATIOS: 70% train / 15% validation / 15% test
    
    WHY SMOTE (Synthetic Minority Oversampling Technique):
      - Only 3.5% of transactions are fraud
      - A model trained on this will always predict "not fraud" and get 96.5% accuracy
      - SMOTE creates SYNTHETIC fraud examples by interpolating between real ones
      - Applied ONLY to training data — never touch validation/test sets
        (test set must reflect real-world distribution to be a valid benchmark)
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]
    
    print(f"\n📊 Original class distribution:")
    print(f"   Not fraud: {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"   Fraud:     {(y==1).sum():,} ({(y==1).mean():.1%})")
    
    # First split off test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Then split remaining into train (82.4% of temp) and val (17.6% of temp)
    # This gives us 70% / 15% / 15% of the original data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    # Apply SMOTE to training data only
    print("\n⚖️  Applying SMOTE to balance training set...")
    # sampling_strategy=0.3 means: make fraud = 30% of non-fraud count
    # We don't go to 50/50 because it's overkill and slows training
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"✅ After SMOTE:")
    print(f"   Train: {len(X_train_bal):,} samples ({y_train_bal.mean():.1%} fraud)")
    print(f"   Val:   {len(X_val):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    return X_train_bal, X_val, X_test, y_train_bal, y_val, y_test, list(feature_cols)


def run_preprocessing_pipeline(data_dir: str = 'data', save_dir: str = 'models'):
    """
    Master function — runs the complete preprocessing pipeline.
    Call this from the command line or your training notebook.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Load
    df = load_data(data_dir)
    
    # Step 2: Feature engineering
    df = engineer_features(df)
    
    # Step 3: Clean + encode
    df, encoders = clean_and_encode(df, fit=True)
    
    # Step 4: Split + balance
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        split_and_balance(df)
    
    # Save encoders so we can use them at inference time
    joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))
    print(f"\n💾 Encoders saved to {save_dir}/encoders.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, encoders


if __name__ == '__main__':
    run_preprocessing_pipeline()
