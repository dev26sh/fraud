from ml.model import FraudDetectionModel
import pandas as pd
import datetime
import time

class MLInterface:
    def __init__(self, model_path='ml/saved_models/fraud_model.pkl'):
        self.model = FraudDetectionModel()
        self.model.load_model(model_path)
    
    def analyze_transaction(self, transaction):
        # Convert blockchain transaction to format suitable for ML model
        ml_transaction = {
            'sender': transaction['sender'],
            'receiver': transaction['receiver'],
            'amount': transaction['amount'],
            'timestamp': datetime.datetime.fromtimestamp(transaction['timestamp'])
        }
        
        # Get prediction
        prediction = self.model.predict(ml_transaction)
        
        return prediction
    
    def process_transactions_batch(self, transactions):
        results = []
        for tx in transactions:
            result = self.analyze_transaction(tx)
            results.append({
                'transaction_id': tx['id'],
                'is_fraud': result['is_fraud'],
                'fraud_probability': result['fraud_probability']
            })
        return results