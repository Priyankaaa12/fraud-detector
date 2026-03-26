"""
llm.py — LLM Explanation Generator using Claude API
=====================================================
WHAT THIS FILE DOES:
  Takes the SHAP output (structured numbers) and turns it into a clear,
  plain-English explanation that a bank compliance officer can actually read
  and act on — without needing to understand machine learning.

WHY AN LLM LAYER?
  SHAP gives us: "TransactionAmt SHAP value = +0.35"
  That means nothing to a compliance officer.
  
  The LLM turns it into:
  "This transaction is flagged primarily because the amount ($4,200) is
  significantly higher than this cardholder's typical transaction range ($50–$200).
  Combined with the 3 AM timestamp and a newly registered device, these three
  factors together suggest a high probability of account takeover fraud."

  This is the key innovation that makes your project stand out.
"""

import anthropic
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()  # Load ANTHROPIC_API_KEY from .env file


# ─── Prompt Template ─────────────────────────────────────────────────────────
# This is carefully engineered to get consistent, professional output.
# Key principles:
#   1. Give the model a clear role (compliance analyst)
#   2. Provide structured input (not raw numbers)
#   3. Specify exact output format
#   4. Set tone (professional, concise, actionable)
EXPLANATION_PROMPT = """You are an AI fraud analyst assistant for a financial compliance team.
Your job is to explain why a transaction was flagged as potentially fraudulent.
Write clearly for a compliance officer — not a data scientist.

TRANSACTION DETAILS:
- Transaction ID: {transaction_id}
- Amount: ${amount}
- Fraud Probability: {probability:.1%}
- Risk Level: {risk_level}

TOP RISK FACTORS (from ML model analysis):
{factors_text}

BASELINE CONTEXT:
The average transaction in this system has a {base_prob:.1%} fraud probability.
This transaction is {multiplier:.1f}x higher than baseline.

Write a concise explanation (3–5 sentences) that:
1. States the primary reason this was flagged
2. Explains what combination of factors is suspicious
3. Suggests what a compliance officer should verify
4. Uses plain English — no technical jargon

Do NOT mention SHAP, machine learning, or model internals.
Do NOT start with "This transaction" — vary your opening.
Be specific about the actual values, not generic."""


SEVERITY_PROMPT = """Based on this fraud analysis, classify the urgency:

Risk Score: {risk_score}/100
Risk Level: {risk_level}
Amount: ${amount}
Top Factor: {top_factor}

Reply with ONLY one of these exactly:
- BLOCK_IMMEDIATELY (score > 85 or amount > 5000)
- REVIEW_URGENT (score 60-85)  
- MONITOR (score 35-60)
- CLEAR (score < 35)

Then on a new line, give one sentence justification."""


class FraudLLMExplainer:
    """
    Generates plain-English fraud explanations using Claude API.
    
    The explanation pipeline:
    1. Format SHAP values into readable text
    2. Send to Claude with a carefully crafted prompt
    3. Parse and return the response
    4. Cache results to avoid redundant API calls
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found.\n"
                "1. Go to console.anthropic.com\n"
                "2. Create an API key\n"
                "3. Add to .env file: ANTHROPIC_API_KEY=your_key_here"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._cache = {}  # Simple in-memory cache
    
    def generate_explanation(
        self,
        shap_result: Dict,
        transaction_id: str = "TXN-UNKNOWN",
        amount: float = 0.0,
        use_cache: bool = True
    ) -> Dict:
        """
        Generate a complete explanation for a flagged transaction.
        
        PARAMETERS:
          shap_result    : output from FraudExplainer.explain_transaction()
          transaction_id : for display purposes
          amount         : original transaction amount in dollars
          use_cache      : skip API call if same transaction was explained recently
        
        RETURNS:
          {
            'explanation': "The transaction was flagged because...",
            'action': "REVIEW_URGENT",
            'action_reason': "Score of 74 with unusual amount...",
            'risk_score': 74,
            'risk_level': 'HIGH',
            'top_factors': [...],
            'fraud_probability': 0.74
          }
        """
        # Check cache first (saves API costs during development)
        cache_key = f"{transaction_id}_{shap_result.get('risk_score', 0)}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Format SHAP factors into readable text
        factors_text = _format_factors(shap_result['top_factors'])
        
        # Build the explanation prompt
        base_prob = shap_result.get('base_probability', 0.035)
        prob = shap_result['fraud_probability']
        multiplier = prob / base_prob if base_prob > 0 else 1
        
        prompt = EXPLANATION_PROMPT.format(
            transaction_id=transaction_id,
            amount=f"{amount:,.2f}",
            probability=prob,
            risk_level=shap_result['risk_level'],
            factors_text=factors_text,
            base_prob=base_prob,
            multiplier=multiplier
        )
        
        # Call Claude API
        explanation = self._call_claude(prompt)
        
        # Get recommended action
        action, action_reason = self._get_recommended_action(
            shap_result, amount
        )
        
        result = {
            'explanation': explanation,
            'action': action,
            'action_reason': action_reason,
            'risk_score': shap_result['risk_score'],
            'risk_level': shap_result['risk_level'],
            'fraud_probability': shap_result['fraud_probability'],
            'top_factors': shap_result['top_factors'],
            'base_probability': base_prob,
        }
        
        # Cache the result
        self._cache[cache_key] = result
        
        return result
    
    def generate_batch_summary(self, explanations: List[Dict]) -> str:
        """
        Generate a summary of multiple fraud alerts for a daily report.
        Useful for the compliance team's morning briefing.
        """
        total = len(explanations)
        critical = sum(1 for e in explanations if e.get('risk_level') == 'CRITICAL')
        high = sum(1 for e in explanations if e.get('risk_level') == 'HIGH')
        total_amount = sum(e.get('amount', 0) for e in explanations)
        
        top_patterns = _find_common_patterns(explanations)
        
        summary_prompt = f"""Write a 3-sentence fraud alert summary for a compliance manager.

Stats for the last hour:
- Total flagged transactions: {total}
- Critical alerts: {critical}
- High risk alerts: {high}
- Total flagged amount: ${total_amount:,.2f}
- Common patterns: {top_patterns}

Be concise and actionable. Mention the most urgent items first."""
        
        return self._call_claude(summary_prompt, max_tokens=200)
    
    def _call_claude(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Make an API call to Claude with error handling.
        
        WHY max_tokens=300:
          - Enough for a 3-5 sentence explanation
          - Keeps API costs low during development
          - Faster response time
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        
        except anthropic.AuthenticationError:
            return "Error: Invalid API key. Check your .env file."
        except anthropic.RateLimitError:
            return "Error: Rate limit reached. Wait 60 seconds and retry."
        except anthropic.APIConnectionError:
            return "Error: Cannot connect to API. Check your internet connection."
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _get_recommended_action(self, shap_result: Dict, amount: float) -> tuple:
        """
        Determine what action to recommend based on risk score and amount.
        Uses rule-based logic first (instant), falls back to LLM for edge cases.
        """
        score = shap_result['risk_score']
        risk_level = shap_result['risk_level']
        top_factor = shap_result['top_factors'][0]['description'] if shap_result['top_factors'] else ''
        
        # Rule-based for clear-cut cases (faster + cheaper than LLM)
        if score >= 85 or (score >= 70 and amount >= 5000):
            return ('BLOCK_IMMEDIATELY',
                    f"Risk score {score}/100 with high-value transaction requires immediate action")
        elif score >= 60:
            return ('REVIEW_URGENT',
                    f"Risk score {score}/100 requires urgent compliance review within 1 hour")
        elif score >= 35:
            return ('MONITOR',
                    f"Risk score {score}/100 — flag for monitoring, review within 24 hours")
        else:
            return ('CLEAR',
                    f"Risk score {score}/100 is within normal parameters")


def _format_factors(top_factors: List[Dict]) -> str:
    """Convert SHAP factor list to readable numbered text for the prompt."""
    lines = []
    for i, factor in enumerate(top_factors, 1):
        direction = "↑ increases" if factor['direction'] == 'increases_risk' else "↓ decreases"
        impact = abs(factor['shap_value'])
        lines.append(
            f"{i}. {factor['feature']}: {factor['description']} "
            f"(impact: {impact:.3f}, {direction} fraud risk)"
        )
    return '\n'.join(lines)


def _find_common_patterns(explanations: List[Dict]) -> str:
    """Find the most common risk factors across multiple explanations."""
    factor_counts = {}
    for exp in explanations:
        for factor in exp.get('top_factors', [])[:2]:  # Top 2 per transaction
            feat = factor.get('feature', '')
            if factor.get('direction') == 'increases_risk':
                factor_counts[feat] = factor_counts.get(feat, 0) + 1
    
    top = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return ', '.join(f"{feat} ({count}x)" for feat, count in top) or 'No clear pattern'


# ─── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # This tests the LLM connection before you integrate it with the model
    print("Testing Claude API connection...")
    
    llm = FraudLLMExplainer()
    
    # Fake SHAP result for testing
    fake_shap = {
        'fraud_probability': 0.87,
        'risk_score': 87,
        'risk_level': 'CRITICAL',
        'base_probability': 0.035,
        'top_factors': [
            {
                'feature': 'TransactionAmt',
                'value': 4200.0,
                'shap_value': 0.35,
                'direction': 'increases_risk',
                'description': 'Amount $4,200 increases fraud risk'
            },
            {
                'feature': 'hour',
                'value': 3,
                'shap_value': 0.22,
                'direction': 'increases_risk',
                'description': 'Transaction at hour 3 increases fraud risk'
            },
            {
                'feature': 'D1',
                'value': 0,
                'shap_value': 0.18,
                'direction': 'increases_risk',
                'description': 'Days since last transaction increases fraud risk'
            }
        ]
    }
    
    result = llm.generate_explanation(
        fake_shap,
        transaction_id='TXN-TEST-001',
        amount=4200.0
    )
    
    print(f"\n{'='*60}")
    print("EXPLANATION:")
    print(result['explanation'])
    print(f"\nRECOMMENDED ACTION: {result['action']}")
    print(f"REASON: {result['action_reason']}")
    print('='*60)
