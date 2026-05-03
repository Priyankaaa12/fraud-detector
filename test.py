import sys
sys.path.insert(0, 'src')
from explain import FraudExplainer
import joblib

model = joblib.load('models/fraud_model.pkl')
fn = joblib.load('models/feature_names.pkl')
exp = FraudExplainer(model, fn)

cases = [
    ('Grocery $45 Tuesday 11am',  {'TransactionAmt':45.0,'hour':11,'day_of_week':1,'D1':14.0,'C1':2,'card4':2,'P_emaildomain':0,'log_amount':3.83,'is_large_amount':0,'C2':2,'C6':1,'V12':1.0,'V13':1.0,'card1':5000,'card2':321,'addr1':299,'addr2':87,'dist1':10.0,'D10':14.0,'D15':30.0,'M4':1,'M5':1,'M6':1}),
    ('Coffee $5 Wednesday 9am',   {'TransactionAmt':5.0,'hour':9,'day_of_week':2,'D1':7.0,'C1':3,'card4':2,'P_emaildomain':0,'log_amount':1.79,'is_large_amount':0,'C2':3,'C6':2,'V12':2.0,'V13':2.0,'card1':5000,'card2':321,'addr1':299,'addr2':87,'dist1':5.0,'D10':7.0,'D15':14.0,'M4':2,'M5':1,'M6':1}),
    ('Subscription $15 Monday 2pm',{'TransactionAmt':15.0,'hour':14,'day_of_week':0,'D1':30.0,'C1':1,'card4':2,'P_emaildomain':0,'log_amount':2.77,'is_large_amount':0,'C2':1,'C6':1,'V12':1.0,'V13':1.0,'card1':7200,'card2':452,'addr1':390,'addr2':60,'dist1':0.0,'D10':30.0,'D15':60.0,'M4':1,'M5':2,'M6':2}),
    ('Utility $120 Thursday 10am',{'TransactionAmt':120.0,'hour':10,'day_of_week':3,'D1':45.0,'C1':4,'card4':1,'P_emaildomain':0,'log_amount':4.79,'is_large_amount':0,'C2':4,'C6':3,'V12':3.0,'V13':3.0,'card1':8500,'card2':512,'addr1':450,'addr2':87,'dist1':0.0,'D10':45.0,'D15':90.0,'M4':2,'M5':2,'M6':2}),
    ('Petrol $60 Friday 8am',     {'TransactionAmt':60.0,'hour':8,'day_of_week':4,'D1':3.0,'C1':5,'card4':2,'P_emaildomain':0,'log_amount':4.11,'is_large_amount':0,'C2':5,'C6':4,'V12':4.0,'V13':4.0,'card1':6000,'card2':400,'addr1':350,'addr2':87,'dist1':2.0,'D10':3.0,'D15':10.0,'M4':2,'M5':2,'M6':2}),
    ('Amazon $30 Wednesday 3pm',  {'TransactionAmt':30.0,'hour':15,'day_of_week':2,'D1':2.0,'C1':6,'card4':1,'P_emaildomain':0,'log_amount':3.43,'is_large_amount':0,'C2':6,'C6':5,'V12':5.0,'V13':5.0,'card1':9000,'card2':600,'addr1':500,'addr2':87,'dist1':0.0,'D10':2.0,'D15':5.0,'M4':2,'M5':2,'M6':2}),
]

print('='*55)
print('TRANSACTION SCORE ANALYSIS')
print('='*55)
for name, t in cases:
    r = exp.explain_transaction(t)
    print(f"{name}")
    print(f"  Score: {r['risk_score']}/100  |  Level: {r['risk_level']}  |  Prob: {r['fraud_probability']:.1%}")
    print()