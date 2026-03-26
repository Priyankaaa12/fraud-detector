"""
models.py — Pydantic Request & Response Models
================================================
WHAT THIS FILE DOES:
  Defines the exact shape of data that goes IN and OUT of the API.
  FastAPI uses these to:
    1. Validate incoming requests (return 422 if invalid)
    2. Auto-generate Swagger docs
    3. Serialize responses to JSON

WHY PYDANTIC:
  Without validation, a typo like sending "TransactionAmt": "four thousand"
  instead of 4000.0 would crash the model. Pydantic catches this before
  the data reaches the ML code.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TransactionInput(BaseModel):
    """
    Input model for a transaction to be evaluated.
    All fields are optional — missing ones default to 0.
    In production you'd make more fields required.
    """
    
    transaction_id: Optional[str] = Field(
        None,
        description="Unique transaction identifier",
        example="TXN-2024-001"
    )
    
    # Core transaction fields
    TransactionAmt: Optional[float] = Field(
        0.0,
        description="Transaction amount in USD",
        example=4200.0,
        ge=0  # Must be >= 0
    )
    ProductCD: Optional[int] = Field(
        0,
        description="Product code (label-encoded)",
        example=4
    )
    
    # Card information
    card1: Optional[float] = Field(0, description="Card identifier")
    card2: Optional[float] = Field(0, description="Card CVV2 info")
    card4: Optional[int] = Field(0, description="Card brand (encoded)")
    card6: Optional[int] = Field(0, description="Card type (encoded)")
    
    # Address
    addr1: Optional[float] = Field(0, description="Billing zip code")
    addr2: Optional[float] = Field(0, description="Billing country")
    dist1: Optional[float] = Field(0, description="Distance metric")
    
    # Email domains (label-encoded)
    P_emaildomain: Optional[int] = Field(0, description="Purchaser email domain")
    R_emaildomain: Optional[int] = Field(0, description="Recipient email domain")
    
    # Count features
    C1: Optional[float] = Field(0, description="Transaction count feature 1")
    C2: Optional[float] = Field(0, description="Transaction count feature 2")
    C6: Optional[float] = Field(0, description="Transaction count feature 6")
    C13: Optional[float] = Field(0, description="Transaction count feature 13")
    C14: Optional[float] = Field(0, description="Transaction count feature 14")
    
    # Time delta features
    D1: Optional[float] = Field(0, description="Days since last transaction")
    D10: Optional[float] = Field(0, description="Days delta feature 10")
    D15: Optional[float] = Field(0, description="Days delta feature 15")
    
    # Match features
    M4: Optional[int] = Field(0, description="Name match flag")
    M5: Optional[int] = Field(0, description="Address match flag")
    M6: Optional[int] = Field(0, description="Phone match flag")
    
    # Vesta features
    V12: Optional[float] = Field(0)
    V13: Optional[float] = Field(0)
    V29: Optional[float] = Field(0)
    V30: Optional[float] = Field(0)
    V33: Optional[float] = Field(0)
    V34: Optional[float] = Field(0)
    
    # Engineered features (computed in preprocess.py)
    hour: Optional[int] = Field(
        12,
        description="Hour of transaction (0-23)",
        example=3,
        ge=0,
        le=23
    )
    day_of_week: Optional[int] = Field(
        1,
        description="Day of week (0=Mon, 6=Sun)",
        example=6,
        ge=0,
        le=6
    )
    log_amount: Optional[float] = Field(
        0,
        description="Natural log of transaction amount"
    )
    is_large_amount: Optional[int] = Field(
        0,
        description="1 if amount > $1000, else 0"
    )
    
    def to_feature_dict(self) -> dict:
        """
        Convert to a plain dict for the SHAP explainer.
        Automatically computes derived features if not provided.
        """
        import math
        d = self.dict(exclude={'transaction_id'})
        
        # Auto-compute log_amount if not provided
        if d.get('log_amount', 0) == 0 and d.get('TransactionAmt', 0) > 0:
            d['log_amount'] = math.log1p(d['TransactionAmt'])
        
        # Auto-compute is_large_amount
        d['is_large_amount'] = 1 if d.get('TransactionAmt', 0) > 1000 else 0
        
        return d
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-2024-001",
                "TransactionAmt": 4200.0,
                "hour": 3,
                "day_of_week": 6,
                "card4": 2,
                "P_emaildomain": 3,
                "D1": 0,
                "C1": 15
            }
        }


class FactorDetail(BaseModel):
    """A single SHAP factor contributing to the fraud score."""
    feature: str = Field(description="Feature name")
    value: Optional[float] = Field(description="Raw feature value")
    shap_value: float = Field(description="SHAP contribution value")
    direction: str = Field(description="'increases_risk' or 'decreases_risk'")
    description: str = Field(description="Human-readable description")


class PredictionResponse(BaseModel):
    """Complete fraud prediction response."""
    
    transaction_id: str
    fraud_probability: float = Field(
        description="Probability of fraud (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    risk_score: int = Field(
        description="Risk score (0 to 100)",
        ge=0,
        le=100
    )
    risk_level: str = Field(
        description="Risk level: LOW, MEDIUM, HIGH, or CRITICAL"
    )
    action: str = Field(
        description="Recommended action: CLEAR, MONITOR, REVIEW_URGENT, BLOCK_IMMEDIATELY"
    )
    action_reason: str = Field(
        description="Brief reason for the recommended action"
    )
    explanation: str = Field(
        description="Plain-English explanation of why this was flagged"
    )
    top_factors: List[FactorDetail] = Field(
        description="Top 5 features contributing to the fraud score"
    )
    base_probability: float = Field(
        description="Baseline fraud probability (average across all transactions)"
    )
    timestamp: str = Field(
        description="ISO timestamp of when prediction was made"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-2024-001",
                "fraud_probability": 0.87,
                "risk_score": 87,
                "risk_level": "CRITICAL",
                "action": "BLOCK_IMMEDIATELY",
                "action_reason": "Risk score 87/100 with high-value transaction",
                "explanation": "This transaction shows multiple high-risk signals...",
                "top_factors": [
                    {
                        "feature": "TransactionAmt",
                        "value": 4200.0,
                        "shap_value": 0.35,
                        "direction": "increases_risk",
                        "description": "Amount $4,200 increases fraud risk"
                    }
                ],
                "base_probability": 0.035,
                "timestamp": "2024-11-15T03:42:17"
            }
        }


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
    llm_available: bool
    timestamp: str
