"""
simulator.py — Transaction Stream Simulator
============================================
WHAT THIS FILE DOES:
  Generates a continuous stream of fake transactions for demo and testing.
  Sends them to the FastAPI endpoint and stores results.
  This is what powers the "live feed" in the dashboard.

WHY A SIMULATOR:
  We can't use real bank data in a portfolio project.
  This generates realistic-looking transactions with:
  - Normal patterns (small amounts, business hours, familiar devices)
  - Fraud patterns (large amounts, 3 AM, new devices, foreign locations)
  - The fraud rate matches real-world (about 3-5%)

HOW TO RUN:
  python src/simulator.py
  (Keep this running in a separate terminal while the dashboard is open)
"""

import requests
import random
import time
import json
from datetime import datetime
import math
import threading


# ─── API Configuration ───────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"
TRANSACTIONS_PER_SECOND = 0.5  # One transaction every 2 seconds (adjustable)
FRAUD_RATE = 0.05  # 5% of generated transactions will be fraud


# ─── Transaction Templates ───────────────────────────────────────────────────
# Normal transaction profile
NORMAL_PROFILE = {
    'amount_range': (5, 500),
    'hour_range': (8, 22),         # Business hours
    'day_range': (0, 4),           # Weekdays
    'd1_range': (30, 365),         # Regular returning customer
    'c1_range': (1, 10),           # Normal transaction count
    'email_domains': [0, 1, 2, 3], # Common email providers
}

# Fraud transaction profile
FRAUD_PROFILE = {
    'amount_range': (500, 15000),   # Larger amounts
    'hour_range': (0, 5),           # Late night / early morning
    'day_range': (5, 6),            # Weekends
    'd1_range': (0, 2),             # Very recent (account takeover)
    'c1_range': (50, 200),          # Unusually high transaction count
    'email_domains': [8, 9, 10],    # Unusual providers
}

# Card types (encoded): Visa=2, Mastercard=1, etc.
CARD_TYPES = [1, 2, 3, 4]
# Product codes: W=0, H=1, C=2, S=3, R=4
PRODUCT_CODES = [0, 1, 2, 3, 4]


def generate_transaction(is_fraud: bool = False, transaction_num: int = 0) -> dict:
    """
    Generate a single synthetic transaction.
    
    PARAMETERS:
      is_fraud       : if True, use fraud patterns; otherwise normal
      transaction_num: used for generating unique IDs
    
    The transaction matches the feature format expected by our model.
    """
    profile = FRAUD_PROFILE if is_fraud else NORMAL_PROFILE
    
    # Generate base amount
    amount = random.uniform(*profile['amount_range'])
    
    # Add some realistic variance
    if is_fraud:
        # Fraudsters often make round-number transactions or very specific amounts
        if random.random() < 0.3:
            amount = round(amount, -2)  # Round to nearest 100
    
    hour = random.randint(*profile['hour_range'])
    day = random.randint(*profile['day_range'])
    d1 = random.uniform(*profile['d1_range'])
    c1 = random.randint(*profile['c1_range'])
    
    transaction = {
        'transaction_id': f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{transaction_num:04d}",
        'TransactionAmt': round(amount, 2),
        'ProductCD': random.choice(PRODUCT_CODES),
        'card1': random.randint(1000, 9999),
        'card2': random.randint(100, 999),
        'card4': random.choice(CARD_TYPES),
        'card6': random.randint(0, 1),
        'addr1': random.randint(100, 600),
        'addr2': random.randint(0, 100),
        'dist1': random.uniform(0, 500) if not is_fraud else random.uniform(1000, 5000),
        'P_emaildomain': random.choice(profile['email_domains']),
        'R_emaildomain': random.choice(profile['email_domains']),
        'C1': c1,
        'C2': random.randint(0, c1),
        'C6': random.randint(0, 20),
        'C13': random.randint(0, 50),
        'C14': random.randint(0, 30),
        'D1': round(d1, 1),
        'D10': round(random.uniform(0, 30), 1),
        'D15': round(random.uniform(0, 60), 1),
        'M4': random.randint(0, 3),
        'M5': random.randint(0, 2),
        'M6': random.randint(0, 2),
        'V12': round(random.uniform(0, 5), 2),
        'V13': round(random.uniform(0, 5), 2),
        'V29': round(random.uniform(0, 3), 2),
        'V30': round(random.uniform(0, 3), 2),
        'V33': round(random.uniform(0, 2), 2),
        'V34': round(random.uniform(0, 2), 2),
        'hour': hour,
        'day_of_week': day,
        'log_amount': round(math.log1p(amount), 4),
        'is_large_amount': 1 if amount > 1000 else 0,
    }
    
    return transaction


def send_transaction(transaction: dict, verbose: bool = True) -> dict:
    """
    Send a transaction to the FastAPI prediction endpoint.
    Returns the full prediction response.
    """
    try:
        response = requests.post(
            API_URL,
            json=transaction,
            timeout=30,
            params={'include_llm': False}  # Skip LLM for speed in simulation
        )
        response.raise_for_status()
        result = response.json()
        
        if verbose:
            risk_icons = {
                'CRITICAL': '🔴',
                'HIGH': '🟠', 
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            icon = risk_icons.get(result.get('risk_level', ''), '⚪')
            print(
                f"{icon} {result['transaction_id']} | "
                f"${transaction['TransactionAmt']:>8.2f} | "
                f"Score: {result['risk_score']:>3}/100 | "
                f"{result['risk_level']:<8} | "
                f"Action: {result['action']}"
            )
        
        return result
    
    except requests.exceptions.ConnectionError:
        if verbose:
            print("❌ Cannot connect to API. Is it running? → uvicorn api.main:app --reload")
        return {}
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return {}


def run_simulation(
    duration_seconds: float = None,
    interval: float = 2.0,
    verbose: bool = True
):
    """
    Run the transaction simulation loop.
    
    PARAMETERS:
      duration_seconds : how long to run (None = run forever until Ctrl+C)
      interval         : seconds between transactions
      verbose          : print each transaction to console
    
    Run this in a separate terminal while the API and dashboard are running.
    """
    print("🚀 Transaction Simulator Starting")
    print(f"   API URL: {API_URL}")
    print(f"   Interval: {interval}s between transactions")
    print(f"   Fraud rate: {FRAUD_RATE:.0%}")
    print("   Press Ctrl+C to stop\n")
    print("-" * 75)
    print(f"{'Status':<4} {'Transaction ID':<30} {'Amount':>10} {'Score':>6} {'Level':<10} Action")
    print("-" * 75)
    
    transaction_num = 0
    start_time = time.time()
    
    try:
        while True:
            # Randomly decide if this is fraud
            is_fraud = random.random() < FRAUD_RATE
            
            # Generate and send transaction
            transaction = generate_transaction(is_fraud, transaction_num)
            result = send_transaction(transaction, verbose)
            
            transaction_num += 1
            
            # Check if we should stop
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                print(f"\n✅ Simulation complete: {transaction_num} transactions sent")
                break
            
            # Wait before next transaction
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Simulation stopped after {transaction_num} transactions")


def generate_batch(n: int = 100, fraud_rate: float = 0.05) -> list:
    """
    Generate a batch of transactions (for testing without the API).
    Returns a list of (transaction_dict, is_fraud) tuples.
    """
    transactions = []
    for i in range(n):
        is_fraud = random.random() < fraud_rate
        t = generate_transaction(is_fraud, i)
        transactions.append((t, is_fraud))
    return transactions


if __name__ == '__main__':
    import sys
    
    # Allow adjusting speed from command line
    # Usage: python src/simulator.py 1.0  (1 second between transactions)
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    run_simulation(interval=interval)
