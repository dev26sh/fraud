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
        self.features = [
            'amount', 'amount_log', 'is_high_amount', 
            'hour', 'is_unusual_hour', 'day_of_week',
            'sender_frequency', 'receiver_frequency',
            'amount_hour_interaction',
            'sender_hash', 'receiver_hash'
        ]
        
    def preprocess(self, df, training=True):
        # Feature engineering
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
        
        # Calculate transaction frequency
        sender_counts = df.groupby('sender').size().reset_index(name='sender_frequency')
        receiver_counts = df.groupby('receiver').size().reset_index(name='receiver_frequency')
        
        df = pd.merge(df, sender_counts, on='sender', how='left')
        df = pd.merge(df, receiver_counts, on='receiver', how='left')
        
        # Add new features to better capture fraud patterns
        df['amount_log'] = np.log1p(df['amount'])  # Log transformation of amount
        df['is_high_amount'] = (df['amount'] > 8000).astype(int)  # Binary flag for high amounts
        df['is_unusual_hour'] = ((df['hour'] < 9) | (df['hour'] > 17)).astype(int)  # Business hours check
        
        # Add interaction features
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        
        # Convert addresses to numerical features (hash)
        df['sender_hash'] = pd.util.hash_array(df['sender'].values) % 10_000_000
        df['receiver_hash'] = pd.util.hash_array(df['receiver'].values) % 10_000_000
        
        # Select features
        X = df[self.features]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        # Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess
        X, y = self.preprocess(df, training=True)
        
        # Get feature names - this is the missing part
        features = [
            'amount', 'amount_log', 'is_high_amount', 
            'hour', 'is_unusual_hour', 'day_of_week',
            'sender_frequency', 'receiver_frequency',
            'amount_hour_interaction',
            'sender_hash', 'receiver_hash'
        ]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model with improved parameters
        self.model = RandomForestClassifier(
            n_estimators=500, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate with more metrics
        from sklearn.metrics import classification_report, roc_auc_score
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("\n===== MODEL EVALUATION =====")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Calculate feature importance - fixed here
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        print("\n===== FEATURE IMPORTANCE =====")
        for feature, importance in feature_importance:
            print(f"{feature}: {importance:.4f}")
        
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
        X, _ = self.preprocess(df, training=False)
        
        # Predict
        fraud_proba = self.model.predict_proba(X)[:, 1]
        fraud_prediction = self.model.predict(X)
        
        return {
            'is_fraud': bool(fraud_prediction[0]),
            'fraud_probability': float(fraud_proba[0])
        }