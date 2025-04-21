import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess(self, df):
        # Feature engineering
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Calculate transaction frequency
        sender_counts = df.groupby('sender').size().reset_index(name='sender_frequency')
        receiver_counts = df.groupby('receiver').size().reset_index(name='receiver_frequency')
        
        df = pd.merge(df, sender_counts, on='sender', how='left')
        df = pd.merge(df, receiver_counts, on='receiver', how='left')
        
        # Convert addresses to numerical features
        df['sender_hash'] = pd.util.hash_array(df['sender'].values)
        df['receiver_hash'] = pd.util.hash_array(df['receiver'].values)
        
        # Select features
        features = [
            'amount', 'hour', 'day_of_week', 
            'sender_frequency', 'receiver_frequency'
        ]
        
        X = df[features]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if y is not None else self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess
        X, y = self.preprocess(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        # Save model
        os.makedirs('ml/saved_models', exist_ok=True)
        with open('ml/saved_models/fraud_model.pkl', 'wb') as f:
            pickle.dump((self.model, self.scaler), f)
    
    def load_model(self, model_path='ml/saved_models/fraud_model.pkl'):
        with open(model_path, 'rb') as f:
            self.model, self.scaler = pickle.load(f)
    
    def predict(self, transaction_data):
        # Convert transaction data to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame(transaction_data)
        
        # Preprocess
        X, _ = self.preprocess(df)
        
        # Predict
        fraud_proba = self.model.predict_proba(X)[:, 1]
        fraud_prediction = self.model.predict(X)
        
        return {
            'is_fraud': bool(fraud_prediction[0]),
            'fraud_probability': float(fraud_proba[0])
        }