"""
train.py — LightGBM Model Training (CPU-Optimised)
====================================================
WHAT THIS FILE DOES:
  Trains a LightGBM gradient boosting model to classify transactions as
  fraudulent or not. Optimised specifically for CPU-only machines with 8GB RAM.

WHY LIGHTGBM (not a neural network):
  1. Faster on CPU — trains in 5-10 minutes vs hours for PyTorch
  2. Better on tabular data — tree-based models beat neural nets on structured
     financial data in most benchmarks
  3. Lower memory — doesn't need GPU VRAM
  4. Industry standard — banks actually use XGBoost/LightGBM in production
  5. Interpretable with SHAP — works perfectly with our explanation layer

EXPECTED RESULT:
  AUC-ROC > 0.93 (industry standard for fraud detection is > 0.90)
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import joblib
import os
import json
from preprocess import run_preprocessing_pipeline


# ─── Model Hyperparameters (CPU-optimised) ───────────────────────────────────
# These are tuned for 8GB RAM, no GPU.
# Key settings:
#   num_leaves: complexity of each tree (lower = faster, less overfitting)
#   n_estimators: number of trees (with early_stopping_rounds, this is a max)
#   learning_rate: how much each tree corrects the previous (lower = better)
#   scale_pos_weight: further helps with class imbalance (fraud:non-fraud ratio)
LGBM_PARAMS = {
    'objective': 'binary',        # Binary classification (fraud / not fraud)
    'metric': 'auc',              # Optimise for AUC during training
    'boosting_type': 'gbdt',      # Gradient Boosting Decision Tree
    'num_leaves': 63,             # Max leaves per tree (good balance for 8GB)
    'max_depth': 8,               # Limit tree depth to prevent overfitting
    'learning_rate': 0.05,        # Small step size = better generalisation
    'n_estimators': 1000,         # Max trees — early stopping will pick best
    'subsample': 0.8,             # Train each tree on 80% of rows (prevents overfit)
    'colsample_bytree': 0.8,      # Train each tree on 80% of features
    'min_child_samples': 20,      # Min samples in a leaf (prevents overfit)
    'scale_pos_weight': 1,        # We already balanced with SMOTE, so keep at 1
    'reg_alpha': 0.1,             # L1 regularisation
    'reg_lambda': 0.1,            # L2 regularisation
    'random_state': 42,
    'n_jobs': -1,                 # Use all CPU cores
    'verbose': -1,                # Suppress noisy output
}


def train_model(X_train, y_train, X_val, y_val, feature_names: list):
    """
    Train LightGBM model with early stopping.
    
    EARLY STOPPING EXPLAINED:
      Instead of training all 1000 trees, we monitor validation AUC after
      each tree. If AUC doesn't improve for 50 rounds, we stop training.
      This prevents overfitting and saves training time.
    
    CALLBACK EXPLAINED:
      We print AUC every 50 rounds so you can watch the model improve.
    """
    print("🚀 Starting model training...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Features: {len(feature_names)}")
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best iteration: {model.best_iteration_}")
    
    return model


def evaluate_model(model, X_val, y_val, X_test, y_test, save_dir: str = 'models'):
    """
    Compute and display all evaluation metrics.
    
    METRICS EXPLAINED:
    
    AUC-ROC (Area Under ROC Curve):
      - Measures how well the model ranks fraud vs non-fraud
      - 1.0 = perfect, 0.5 = random guessing
      - Target: > 0.93 for this dataset
    
    Precision:
      - Of all transactions flagged as fraud, what % actually were fraud?
      - Low precision = too many false alarms (annoying for compliance team)
    
    Recall (Sensitivity):
      - Of all actual frauds, what % did we catch?
      - Low recall = missing real fraud (costly for the bank)
    
    F1 Score:
      - Harmonic mean of precision and recall
      - Useful when you need to balance both concerns
    
    Average Precision:
      - Better than AUC when classes are imbalanced
      - Focus on how well we rank fraud cases at the top
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get probability scores (not just 0/1 predictions)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Convert probabilities to binary predictions at threshold 0.5
    val_preds = (val_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Compute metrics
    val_auc = roc_auc_score(y_val, val_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    val_ap = average_precision_score(y_val, val_probs)
    test_ap = average_precision_score(y_test, test_probs)
    
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nValidation Set:")
    print(f"  AUC-ROC:           {val_auc:.4f}")
    print(f"  Average Precision: {val_ap:.4f}")
    print(f"\nTest Set:")
    print(f"  AUC-ROC:           {test_auc:.4f}")
    print(f"  Average Precision: {test_ap:.4f}")
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, test_preds,
                                target_names=['Not Fraud', 'Fraud']))
    
    # Save metrics to JSON for later reference
    metrics = {
        'val_auc': round(val_auc, 4),
        'test_auc': round(test_auc, 4),
        'val_avg_precision': round(val_ap, 4),
        'test_avg_precision': round(test_ap, 4),
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save confusion matrix
    _plot_confusion_matrix(y_test, test_preds, save_dir)
    
    return metrics


def _plot_confusion_matrix(y_true, y_pred, save_dir: str):
    """Save confusion matrix as PNG for the README / portfolio."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNot Fraud', 'Predicted\nFraud'])
    ax.set_yticklabels(['Actual\nNot Fraud', 'Actual\nFraud'])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}',
                   ha='center', va='center',
                   color='white' if cm[i,j] > cm.max()/2 else 'black',
                   fontsize=14, fontweight='bold')
    
    ax.set_title('Confusion Matrix — Test Set', pad=15, fontsize=13)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Confusion matrix saved to {save_dir}/confusion_matrix.png")


def save_model(model, feature_names: list, save_dir: str = 'models'):
    """
    Save the trained model and feature names.
    
    WHY SAVE FEATURE NAMES:
      At inference time (when a new transaction arrives), the input must
      have EXACTLY the same features in EXACTLY the same order as training.
      We save feature_names so we can enforce this.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(save_dir, 'fraud_model.pkl'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))
    
    print(f"💾 Model saved to {save_dir}/fraud_model.pkl")
    print(f"💾 Feature names saved to {save_dir}/feature_names.pkl")


def run_training_pipeline():
    """
    Master function — runs the complete training pipeline.
    Run this file directly: python src/train.py
    """
    # Step 1: Get preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, _ = \
        run_preprocessing_pipeline()
    
    # Step 2: Train model
    model = train_model(X_train, y_train, X_val, y_val, feature_names)
    
    # Step 3: Evaluate
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    
    # Step 4: Save
    save_model(model, feature_names)
    
    print(f"\n🎉 Training pipeline complete! AUC-ROC: {metrics['test_auc']:.4f}")
    
    return model, feature_names


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run_training_pipeline()
