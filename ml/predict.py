from model import FraudDetectionModel
import json
import sys

def predict_fraud(transaction_json):
    # Load the trained model
    model = FraudDetectionModel()
    model.load_model()
    
    # Parse transaction data
    transaction = json.loads(transaction_json)
    
    # Make prediction
    result = model.predict(transaction)
    
    return result

if __name__ == "__main__":
    # Read transaction data from command line argument
    transaction_json = sys.argv[1]
    
    # Get prediction
    result = predict_fraud(transaction_json)
    
    # Print result as JSON
    print(json.dumps(result))