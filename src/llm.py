import os
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

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


class FraudLLMExplainer:

    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
                "1. Go to console.groq.com\n"
                "2. Create a free API key\n"
                "3. Add to .env file: GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"  # Free, fast model
        self._cache = {}

    def generate_explanation(
        self,
        shap_result: Dict,
        transaction_id: str = "TXN-UNKNOWN",
        amount: float = 0.0,
        use_cache: bool = True
    ) -> Dict:
        cache_key = f"{transaction_id}_{shap_result.get('risk_score', 0)}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        factors_text = _format_factors(shap_result['top_factors'])
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

        explanation = self._call_groq(prompt)
        action, action_reason = self._get_recommended_action(shap_result, amount)

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

        self._cache[cache_key] = result
        return result

    def generate_batch_summary(self, explanations: List[Dict]) -> str:
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

        return self._call_groq(summary_prompt, max_tokens=200)

    def _call_groq(self, prompt: str, max_tokens: int = 300) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def _get_recommended_action(self, shap_result: Dict, amount: float) -> tuple:
        score = shap_result['risk_score']
        factors = shap_result['top_factors']
        top_factor = ''
        if factors:
            f = factors[0]
            top_factor = f.description if hasattr(f, 'description') else f.get('description', '')

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


def _format_factors(top_factors) -> str:
    lines = []
    for i, factor in enumerate(top_factors, 1):
        if hasattr(factor, 'direction'):
            direction = "↑ increases" if factor.direction == 'increases_risk' else "↓ decreases"
            impact = abs(factor.shap_value)
            lines.append(f"{i}. {factor.feature}: {factor.description} (impact: {impact:.3f}, {direction} fraud risk)")
        else:
            direction = "↑ increases" if factor.get('direction') == 'increases_risk' else "↓ decreases"
            impact = abs(factor.get('shap_value', 0))
            lines.append(f"{i}. {factor.get('feature')}: {factor.get('description')} (impact: {impact:.3f}, {direction} fraud risk)")
    return '\n'.join(lines)


def _find_common_patterns(explanations: List[Dict]) -> str:
    factor_counts = {}
    for exp in explanations:
        for factor in exp.get('top_factors', [])[:2]:
            feat = factor.get('feature', '') if isinstance(factor, dict) else factor.feature
            direction = factor.get('direction', '') if isinstance(factor, dict) else factor.direction
            if direction == 'increases_risk':
                factor_counts[feat] = factor_counts.get(feat, 0) + 1
    top = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return ', '.join(f"{feat} ({count}x)" for feat, count in top) or 'No clear pattern'


if __name__ == '__main__':
    print("Testing Groq API connection...")
    llm = FraudLLMExplainer()
    fake_shap = {
        'fraud_probability': 0.87,
        'risk_score': 87,
        'risk_level': 'CRITICAL',
        'base_probability': 0.035,
        'top_factors': [
            {'feature': 'TransactionAmt', 'value': 4200.0, 'shap_value': 0.35,
             'direction': 'increases_risk', 'description': 'Amount $4,200 increases fraud risk'},
            {'feature': 'hour', 'value': 3, 'shap_value': 0.22,
             'direction': 'increases_risk', 'description': 'Transaction at hour 3 increases fraud risk'},
        ]
    }
    result = llm.generate_explanation(fake_shap, transaction_id='TXN-TEST-001', amount=4200.0)
    print(f"\n{'='*60}")
    print("EXPLANATION:")
    print(result['explanation'])
    print(f"\nRECOMMENDED ACTION: {result['action']}")
    print(f"REASON: {result['action_reason']}")
    print('='*60)