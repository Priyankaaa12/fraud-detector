"""
explain.py — SHAP Explainability Engine
========================================
WHAT THIS FILE DOES:
  Takes a trained LightGBM model and a transaction, computes SHAP values,
  and returns a structured explanation: which features pushed the prediction
  towards fraud and by how much.

WHAT IS SHAP?
  SHAP (SHapley Additive exPlanations) is a method to explain any ML model's
  predictions by computing how much each feature contributed to the final score.

  Imagine the model's base prediction is 0.1 (10% fraud chance).
  After seeing a transaction, SHAP tells us:
    +0.35  TransactionAmt = $4,200 (much higher than usual)
    +0.22  hour = 3 AM (suspicious time)
    +0.18  new_device = True
    -0.05  P_emaildomain = gmail.com (common, less suspicious)
    ─────
    = 0.80  Final fraud probability

  This is EXACTLY what a compliance officer needs to make a decision.

WHY SHAP OVER LIME OR OTHER METHODS:
  - SHAP has theoretical guarantees (based on game theory)
  - TreeSHAP (for LightGBM) is very fast — milliseconds per prediction
  - Consistent: same input always gives same SHAP values
  - Works globally (whole model) and locally (single prediction)
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import joblib
import os
import json
from typing import Dict, List, Tuple


class FraudExplainer:
    """
    Wrapper around SHAP TreeExplainer for fraud detection.
    
    Usage:
        explainer = FraudExplainer.load('models/')
        explanation = explainer.explain_transaction(transaction_dict)
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        print("⚙️  Building SHAP TreeExplainer...")
        # TreeExplainer is the fast, exact version for tree-based models
        # It uses a dynamic programming algorithm instead of sampling
        self.shap_explainer = shap.TreeExplainer(
            model,
            feature_perturbation='tree_path_dependent'
        )
        raw_ev = self.shap_explainer.expected_value
        if isinstance(raw_ev, (list, np.ndarray)):
            arr = np.array(raw_ev).flatten()
            self.expected_value = float(arr[1]) if len(arr) > 1 else float(arr[0])
        else:
            self.expected_value = float(raw_ev)
        print("✅ SHAP explainer ready")
    
    def explain_transaction(
        self,
        transaction: Dict,
        top_n: int = 5
    ) -> Dict:
        """
        Given a transaction dict, return a structured SHAP explanation.
        
        PARAMETERS:
          transaction : dict with feature names as keys
          top_n       : how many top features to return
        
        RETURNS:
          {
            'fraud_probability': 0.87,
            'risk_level': 'HIGH',
            'risk_score': 87,
            'base_probability': 0.04,
            'top_factors': [
              {
                'feature': 'TransactionAmt',
                'value': 4200.0,
                'shap_value': 0.35,
                'direction': 'increases_risk',
                'description': 'Transaction amount is unusually high'
              },
              ...
            ],
            'all_shap_values': {...}
          }
        """
        # Build a single-row DataFrame with correct feature order
        df = self._transaction_to_df(transaction)
        
        # Get fraud probability
        prob = float(self.model.predict_proba(df)[0, 1])
        
        # Compute SHAP values for this single transaction
        # shap_values shape: (1, n_features) for binary class 1 (fraud)
        shap_values = self.shap_explainer.shap_values(df)
        
        # Handle both old and new SHAP API formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # class 1 (fraud), row 0
        else:
            shap_vals = shap_values[0]
        
        # Build feature contribution dict
        contributions = {
            feat: float(val)
            for feat, val in zip(self.feature_names, shap_vals)
        }
        
        # Sort by absolute SHAP value (biggest impact first)
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Build top factors with human-readable descriptions
        top_factors = []
        for feature, shap_val in sorted_contributions[:top_n]:
            raw_value = float(df[feature].iloc[0]) if feature in df.columns else None
            top_factors.append({
                'feature': feature,
                'value': raw_value,
                'shap_value': round(shap_val, 4),
                'direction': 'increases_risk' if shap_val > 0 else 'decreases_risk',
                'description': _get_feature_description(feature, raw_value, shap_val)
            })
        
        return {
            'fraud_probability': round(prob, 4),
            'risk_score': int(prob * 100),
            'risk_level': _get_risk_level(prob),
            'base_probability': round(float(self.expected_value), 4),
            'top_factors': top_factors,
            'all_shap_values': {k: round(v, 4) for k, v in contributions.items()},
            'feature_count': len(self.feature_names)
        }
    
    def explain_batch(self, transactions_df: pd.DataFrame) -> List[Dict]:
        """
        Explain multiple transactions at once.
        Useful for dashboards showing recent alerts.
        """
        results = []
        for _, row in transactions_df.iterrows():
            results.append(self.explain_transaction(row.to_dict()))
        return results
    
    def plot_waterfall(
        self,
        transaction: Dict,
        save_path: str = None,
        show: bool = False
    ) -> str:
        """
        Create a SHAP waterfall chart showing how each feature pushes the
        prediction up or down from the baseline.
        
        This is the chart you'll embed in the dashboard's detail view.
        """
        df = self._transaction_to_df(transaction)
        shap_values = self.shap_explainer.shap_values(df)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        
        # Create SHAP Explanation object for plotting
        explanation = shap.Explanation(
            values=sv,
            base_values=self.expected_value,
            data=df.iloc[0].values,
            feature_names=self.feature_names
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            return save_path
        
        if show:
            plt.show()
        
        plt.close()
        return save_path or ''
    
    def plot_summary(self, X_sample: pd.DataFrame, save_path: str = None):
        """
        Global feature importance plot — shows which features matter most
        across all transactions. Use this in your README / portfolio.
        """
        print("📊 Computing SHAP summary (this takes ~30 seconds)...")
        
        # Sample 500 rows to keep it fast
        if len(X_sample) > 500:
            X_sample = X_sample.sample(500, random_state=42)
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"📈 Summary plot saved to {save_path}")
        
        plt.close()
    
    def _transaction_to_df(self, transaction: Dict) -> pd.DataFrame:
        """
        Convert a transaction dict to a DataFrame with exact feature order.
        Missing features are filled with 0 (the model handles this gracefully).
        """
        row = {feat: transaction.get(feat, 0) for feat in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)
    
    def save(self, save_dir: str = 'models'):
        """Save the explainer to disk."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self, os.path.join(save_dir, 'explainer.pkl'))
        print(f"💾 Explainer saved to {save_dir}/explainer.pkl")
    
    @classmethod
    def load(cls, save_dir: str = 'models') -> 'FraudExplainer':
        """Load explainer from disk."""
        path = os.path.join(save_dir, 'explainer.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No explainer found at {path}. Run train.py first."
            )
        return joblib.load(path)


def _get_risk_level(prob: float) -> str:
    """Convert probability to human-readable risk level."""
    if prob >= 0.80:
        return 'CRITICAL'
    elif prob >= 0.60:
        return 'HIGH'
    elif prob >= 0.35:
        return 'MEDIUM'
    else:
        return 'LOW'


def _get_feature_description(feature: str, value, shap_val: float) -> str:
    """
    Generate a human-readable description of why a feature is suspicious.
    This feeds into the LLM prompt for richer explanations.
    """
    direction = "increases" if shap_val > 0 else "decreases"
    
    descriptions = {
        'TransactionAmt': f"Amount ${value:,.2f} {direction} fraud risk",
        'log_amount': f"Log-scaled amount {direction} fraud risk",
        'hour': f"Transaction at hour {int(value) if value else '?'} {direction} fraud risk",
        'day_of_week': f"Day of week pattern {direction} fraud risk",
        'is_large_amount': "Large amount flag" + (" raised" if shap_val > 0 else " not raised"),
        'card1': f"Card identifier pattern {direction} fraud risk",
        'card4': f"Card brand pattern {direction} fraud risk",
        'P_emaildomain': f"Purchaser email domain {direction} fraud risk",
        'R_emaildomain': f"Recipient email domain {direction} fraud risk",
        'dist1': f"Distance between buyer/seller {direction} fraud risk",
        'D1': f"Days since last transaction {direction} fraud risk",
        'C1': f"Transaction count pattern {direction} fraud risk",
    }
    
    return descriptions.get(feature, f"Feature '{feature}' {direction} fraud risk")


def build_explainer_from_model(model_dir: str = 'models') -> FraudExplainer:
    """
    Helper: load a saved model and build the explainer.
    Run after train.py has finished.
    """
    model = joblib.load(os.path.join(model_dir, 'fraud_model.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    explainer = FraudExplainer(model, feature_names)
    explainer.save(model_dir)
    
    return explainer


if __name__ == '__main__':
    print("Building SHAP explainer from saved model...")
    explainer = build_explainer_from_model()
    
    # Quick test with a fake transaction
    test_transaction = {
        'TransactionAmt': 4200.0,
        'hour': 3,
        'day_of_week': 6,
        'is_large_amount': 1,
        'log_amount': 8.34,
        'card1': 1234,
        'card2': 567,
        'card4': 2,   # encoded category
        'card6': 1,
        'P_emaildomain': 3,   # encoded category
        'C1': 15, 'C2': 2,
        'D1': 0,   # 0 days since last transaction (suspicious!)
    }
    
    result = explainer.explain_transaction(test_transaction)
    print(f"\nTest transaction:")
    print(f"  Fraud probability: {result['fraud_probability']:.1%}")
    print(f"  Risk level: {result['risk_level']}")
    print(f"  Top factor: {result['top_factors'][0]['description']}")