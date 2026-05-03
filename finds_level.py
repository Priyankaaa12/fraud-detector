import sys, math
sys.path.insert(0, 'src')
from explain import FraudExplainer
import joblib

model = joblib.load('models/fraud_model.pkl')
fn = joblib.load('models/feature_names.pkl')
exp = FraudExplainer(model, fn)

def make_transaction(amount, hour, day, d1, c1):
    log_amt = math.log1p(amount)
    c2  = max(1, c1 - 1)
    c6  = max(1, c1 // 3)
    c13 = max(1, c1 // 2)
    c14 = max(1, c1 // 4)
    d10 = max(1.0, d1 * 0.8)
    d15 = max(1.0, d1 * 1.2)
    v12 = 1.0 if d1 > 5 else 0.0
    v13 = 1.0 if d1 > 5 else 0.0
    v33 = 1.0 if c1 < 10 else 0.0
    v34 = 1.0 if c1 < 10 else 0.0
    m5  = 2 if d1 > 5 else 1
    m6  = 2 if d1 > 5 else 1
    return {
        'TransactionAmt': amount, 'hour': hour, 'day_of_week': day,
        'D1': d1, 'D10': d10, 'D15': d15,
        'C1': c1, 'C2': c2, 'C6': c6, 'C13': c13, 'C14': c14,
        'card4': 2, 'card1': 7200.0, 'card2': 452.0, 'card6': 1,
        'P_emaildomain': 0, 'R_emaildomain': 0,
        'M4': 2, 'M5': m5, 'M6': m6,
        'V12': v12, 'V13': v13, 'V29': 0.0, 'V30': 0.0, 'V33': v33, 'V34': v34,
        'addr1': 299.0, 'addr2': 87.0, 'dist1': 0.0 if d1 > 10 else 10.0,
        'log_amount': log_amt, 'is_large_amount': 1 if amount > 1000 else 0,
    }

print('='*60)
print('FINDING EXACT VALUES FOR ALL 4 RISK LEVELS')
print('='*60)

# Test many combinations
test_cases = [
    # (label, amount, hour, day, d1, c1)
    # day: 0=Mon,1=Tue,2=Wed,3=Thu,4=Fri,5=Sat,6=Sun
    ('Test A', 15,   14, 0, 30, 1),
    ('Test B', 15,   14, 0, 60, 1),
    ('Test C', 10,   11, 1, 45, 1),
    ('Test D', 25,   10, 2, 30, 2),
    ('Test E', 50,   14, 3, 14, 3),
    ('Test F', 15,    3, 0, 30, 1),
    ('Test G', 15,   14, 5, 30, 1),
    ('Test H', 200,  14, 0, 30, 1),
    ('Test I', 500,  20, 4, 2,  15),
    ('Test J', 8500,  3, 5, 0,  50),
    ('Test K', 15,   14, 0, 90, 1),
    ('Test L', 10,   12, 1, 90, 1),
    ('Test M', 5,    10, 0, 60, 1),
    ('Test N', 20,   16, 3, 45, 2),
    ('Test O', 100,  14, 0, 30, 2),
]

days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

for label, amt, hr, day, d1, c1 in test_cases:
    t = make_transaction(amt, hr, day, d1, c1)
    r = exp.explain_transaction(t)
    print(f"{label}: ${amt:>5} | {hr:>2}h | {days[day]} | D1={d1:>3} | C1={c1:>2} => Score:{r['risk_score']:>3}/100 | {r['risk_level']}")

print()
print('='*60)
print('SUMMARY - USE THESE IN YOUR DEMO:')
print('='*60)