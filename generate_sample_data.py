import pandas as pd
import numpy as np
import random
import datetime
import os

def generate_sample_transactions(n_samples=1000):
    # Create random addresses
    addresses = [
        f"0x{random.randint(0, 2**160):040x}" for _ in range(100)
    ]
    
    # Generate data
    data = {
        'sender': [random.choice(addresses) for _ in range(n_samples)],
        'receiver': [random.choice(addresses) for _ in range(n_samples)],
        'amount': [random.uniform(10, 10000) for _ in range(n_samples)],
        'timestamp': [
            (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S') 
            for _ in range(n_samples)
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add fraud labels
    # Transactions with very high amounts have higher chance of being fraudulent
    high_amount = df['amount'] > 8000
    unusual_time = [int(ts.split()[1].split(':')[0]) not in range(9, 18) for ts in df['timestamp']]
    
    # Combine features to determine fraud probability
    fraud_prob = np.zeros(n_samples)
    fraud_prob = np.where(high_amount, fraud_prob + 0.4, fraud_prob)
    fraud_prob = np.where(unusual_time, fraud_prob + 0.3, fraud_prob)
    
    # Add some randomness
    fraud_prob += np.random.uniform(-0.1, 0.1, n_samples)
    fraud_prob = np.clip(fraud_prob, 0.01, 0.99)
    
    # Generate binary labels
    df['is_fraud'] = np.random.binomial(1, fraud_prob)
    
    # Ensure at least 10% of transactions are fraudulent for balanced training
    if df['is_fraud'].mean() < 0.1:
        idx = np.random.choice(df[df['is_fraud'] == 0].index, 
                              int(0.1 * n_samples) - df['is_fraud'].sum(),
                              replace=False)
        df.loc[idx, 'is_fraud'] = 1
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_transactions.csv', index=False)
    
    print(f"Created sample data with {n_samples} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    return df

if __name__ == "__main__":
    generate_sample_transactions()