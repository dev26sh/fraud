from model import FraudDetectionModel
import pandas as pd
import sys

def train_model(data_path='data/sample_transactions.csv'):
    print(f"Training fraud detection model using data from {data_path}")
    
    # Create and train the model
    model = FraudDetectionModel()
    model.train(data_path)
    
    print("Training completed. Model saved to ml/saved_models/fraud_model.pkl")

if __name__ == "__main__":
    # Use command line argument for data path if provided
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_transactions.csv'
    train_model(data_path)