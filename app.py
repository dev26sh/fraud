from integration.blockchain_interface import BlockchainInterface
from integration.ml_interface import MLInterface
from web3 import Web3
import time
import datetime
import json
import argparse
import pandas as pd
import os
import traceback

def generate_sample_transaction():
    """Generate a sample transaction from the sample_transactions.csv file"""
    # Path to the sample transactions CSV file
    csv_path = os.path.join('data', 'sample_transactions.csv')
    
    # Read the sample transactions if we haven't already
    if not hasattr(generate_sample_transaction, 'transactions'):
        generate_sample_transaction.transactions = pd.read_csv(csv_path)
        print(f"Loaded {len(generate_sample_transaction.transactions)} sample transactions from {csv_path}")
        generate_sample_transaction.current_index = 0
    
    # Get the current transaction
    transaction = generate_sample_transaction.transactions.iloc[generate_sample_transaction.current_index]
    generate_sample_transaction.current_index += 1
    
    # Convert timestamp string to Unix timestamp
    timestamp_str = transaction['timestamp']
    timestamp_dt = pd.to_datetime(timestamp_str)
    timestamp_unix = int(timestamp_dt.timestamp())
    
    # Format the transaction for our use - using column names
    return {
        'sender': transaction['sender'],  
        'receiver': transaction['receiver'],
        'amount': float(transaction['amount']),
        'timestamp': timestamp_unix,  # Now a proper UNIX timestamp
        'is_fraud': int(transaction['is_fraud'])
    }

def main(mode='monitor'):
    # Initialize interfaces
    blockchain = BlockchainInterface()
    ml = MLInterface()
    
    if mode == 'test':
        print("Running in test mode: analyzing sample transactions from CSV file")
        
        # Load all transactions from the CSV
        csv_path = os.path.join('data', 'sample_transactions.csv')
        transactions_df = pd.read_csv(csv_path)
        total_transactions = len(transactions_df)
        
        # Find all fraud indices first for better visibility
        fraud_indices = []
        for i in range(len(transactions_df)):
            if transactions_df.iloc[i]['is_fraud'] == 1:
                fraud_indices.append(i)
                
        print(f"Processing CSV with {total_transactions} total transactions from {csv_path}")
        print(f"Found {len(fraud_indices)} fraudulent transactions at indices: {fraud_indices}")
        print("Analyzing only fraudulent transactions...\n")
        
        # Track fraudulent transactions and ML performance
        fraud_count = 0
        processed_count = 0
        ml_correct = 0
        ml_missed = 0
        avg_fraud_score = 0
        
        # Process each transaction in the CSV
        for i in range(total_transactions):
            try:
                # Generate sample transaction from CSV
                tx = generate_sample_transaction()
                current_index = generate_sample_transaction.current_index - 1
                processed_count += 1
                
                # Skip non-fraudulent transactions
                if tx['is_fraud'] == 0:
                    continue
                
                fraud_count += 1
                
                # Convert float amount to integer for blockchain
                int_amount = int(tx['amount'])
                
                # Convert receiver address to checksum format
                checksummed_receiver = Web3.to_checksum_address(tx['receiver'])
                
                try:
                    # Add transaction to blockchain with better error handling
                    tx_id = blockchain.add_transaction(checksummed_receiver, int_amount)
                    print(f"FRAUDULENT TRANSACTION #{fraud_count}")
                    print(f"CSV Index: {current_index}")
                    print(f"Transaction ID: {tx_id}")
                    print(f"Amount: {tx['amount']} to {tx['receiver']}")
                    print(f"Original timestamp: {datetime.datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    try:
                        # Get transaction from blockchain with better error handling
                        blockchain_tx = blockchain.get_transaction(tx_id)
                        
                        # Make sure the transaction has the expected format for ML analysis
                        # Create a compatible transaction object for the ML model
                        ml_tx = {
                            'sender': tx['sender'],  # Use original sender since blockchain returns msg.sender
                            'receiver': tx['receiver'],
                            'amount': tx['amount'],
                            'timestamp': tx['timestamp']
                        }
                        
                        # Analyze transaction
                        prediction = ml.analyze_transaction(ml_tx)
                        print(f"ML Analysis: {prediction}")
                        
                        # Track ML performance
                        fraud_prob = prediction.get('fraud_probability', 0)
                        avg_fraud_score += fraud_prob
                        
                        # Check if ML agrees with CSV labeling
                        if fraud_prob >= 0.5:
                            blockchain.flag_transaction(tx_id, fraud_prob)
                            print(f"✓ ML AGREES: Transaction {tx_id} flagged as potentially fraudulent")
                            ml_correct += 1
                        else:
                            print(f"⚠️ ML DISAGREES: CSV says fraud, but ML gives only {fraud_prob:.2f} probability")
                            print(f"   Features: Amount={tx['amount']}, Hour={datetime.datetime.fromtimestamp(tx['timestamp']).hour}")
                            ml_missed += 1
                            
                    except Exception as e:
                        print(f"Error processing blockchain transaction: {e}")
                        traceback.print_exc(limit=1)
                    
                except Exception as e:
                    print(f"Error with blockchain transaction: {e}")
                    traceback.print_exc(limit=1)
                
                print("-" * 60)
                time.sleep(1)  # Small delay between transactions
            
            except Exception as e:
                print(f"Error processing transaction {i+1}: {e}")
                continue
        
        # Calculate ML performance metrics
        if fraud_count > 0:
            avg_fraud_score = avg_fraud_score / fraud_count
            recall = ml_correct / fraud_count if fraud_count > 0 else 0
            
            print("\n===== ML MODEL PERFORMANCE =====")
            print(f"ML Detection Rate: {recall:.2f} ({ml_correct} detected out of {fraud_count} fraudulent transactions)")
            print(f"Average fraud probability score: {avg_fraud_score:.2f}")
            print(f"Missed frauds: {ml_missed}")
            print("===============================\n")
            
        print(f"Summary: Found {fraud_count} fraudulent transactions in CSV of {total_transactions} total)")
            
    elif mode == 'monitor':
        # Rest of the monitor mode code remains unchanged
        print("Running in monitoring mode: continuously checking for new transactions")
        
        last_checked_id = 0
        while True:
            try:
                # Get the latest transaction count
                transaction_count = blockchain.contract.functions.transactionCount().call()
                
                # Check for new transactions
                if transaction_count > last_checked_id:
                    print(f"Found {transaction_count - last_checked_id} new transactions")
                    
                    # Analyze each new transaction
                    for tx_id in range(last_checked_id + 1, transaction_count + 1):
                        blockchain_tx = blockchain.get_transaction(tx_id)
                        prediction = ml.analyze_transaction(blockchain_tx)
                        
                        print(f"Transaction {tx_id}: Fraud probability {prediction['fraud_probability']}")
                        
                        # Flag if potentially fraudulent
                        if prediction['fraud_probability'] > 0.7:
                            blockchain.flag_transaction(tx_id, prediction['fraud_probability'])
                            print(f"Transaction {tx_id} flagged as potentially fraudulent")
                    
                    last_checked_id = transaction_count
                
                time.sleep(5)  # Check for new transactions every 5 seconds
                
            except KeyboardInterrupt:
                print("Monitoring stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)  # Wait longer in case of errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized Fraud Detection System")
    parser.add_argument('--mode', choices=['monitor', 'test'], default='monitor',
                       help='Operating mode: monitor (continuous) or test (sample transactions)')
    
    args = parser.parse_args()
    main(args.mode)