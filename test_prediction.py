import sys
sys.path.insert(0, 'backend')
import joblib
import re

# Load models
tfidf = joblib.load('models/tfidf.joblib')
nb = joblib.load('models/naive_bayes.joblib')

# Test with dataset-like messages
test_messages = [
    'I need help cancelling order 12345',
    'How can I get a refund for my purchase',
    'Where is my order',
    'I want to track my refund',
    'Can you help me change my shipping address',
    'I want to cancel my order number 12345',
    'What is your refund policy',
    'How do I check my invoice',
    'I have a complaint about my order',
    'I need to contact customer service'
]

print('Testing with dataset-like messages...')
print('='*60)
for msg in test_messages:
    cleaned = msg.lower().strip()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    vectorized = tfidf.transform([cleaned])
    intent = nb.predict(vectorized)[0]
    proba = nb.predict_proba(vectorized).max()
    print(f'  "{msg}"')
    print(f'    -> Intent: {intent} (confidence: {proba:.4f})')
    print()
