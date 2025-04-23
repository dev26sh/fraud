from integration.blockchain_interface import BlockchainInterface
from web3 import Web3
import random
import argparse
import time
import datetime
import csv
import os

def create_blockchain_transaction():
    """Create a transaction directly on the blockchain with random fraud status"""
    blockchain = BlockchainInterface()
    
    # Randomly determine if this transaction should be fraudulent 
    is_fraud = random.random() < 0.9
    
    # Generate transaction details
    sender_address = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    receiver_address = Web3.to_checksum_address('0x' + ''.join(random.choices('0123456789abcdef', k=40)))
    
    # Generate amount based on fraud criteria from generate_sample_data.py
    # Fraudulent transactions often have high amounts (>8000)
    if is_fraud:
        amount = random.randint(8000, 10000)
    else:
        amount = random.randint(100, 5000)
    
    # Generate timestamp
    now = datetime.datetime.now()
    if is_fraud:
        hour = random.choice(list(range(0, 8)) + list(range(18, 24)))
    else:
        hour = random.randint(9, 17)
    minute = random.randint(0, 59)
    timestamp = now.replace(hour=hour, minute=minute)
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*60)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BLOCKCHAIN TRANSACTION")
    print("-"*60)
    print(f"Sending amount: {amount}")
    print(f"To address: {receiver_address}")
    print(f"Type: {'🚨 FRAUDULENT PATTERN' if is_fraud else '✓ Normal transaction'}")
    print("-"*60)
    
    try:
        tx_id = blockchain.add_transaction(receiver_address, amount)
        print(f"STATUS: ✅ TRANSACTION SUCCESSFUL")
        print(f"Transaction ID: {tx_id}")
        print(f"Block confirmation time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Add transaction to CSV file
        csv_path = os.path.join('data', 'sample_transactions.csv')
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([sender_address, receiver_address, amount, timestamp_str, 1 if is_fraud else 0])
            print(f"✅ Transaction also recorded in CSV file")
        except Exception as e:
            print(f"❌ Error adding to CSV: {e}")
        
        print("="*60 + "\n")
        return tx_id, is_fraud
        
    except Exception as e:
        print(f"STATUS: ❌ TRANSACTION FAILED")
        print(f"Error details: {e}")
        print("="*60 + "\n")
        return None, is_fraud

def append_to_csv(is_fraud=False):
    """Add a new transaction to the CSV file"""
    csv_path = os.path.join('data', 'sample_transactions.csv')
    
    # Generate addresses
    sender = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    receiver = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    
    # Generate amount - high for fraud
    if is_fraud:
        amount = random.uniform(8000, 10000)  # High amount (suspicious)
    else:
        amount = random.uniform(100, 5000)  # Normal amount
    
    # Generate timestamp - unusual hour for fraud
    now = datetime.datetime.now()
    
    if is_fraud:
        # Outside business hours (suspicious)
        hour = random.choice(list(range(0, 8)) + list(range(18, 24)))
        minute = random.randint(0, 59)
        timestamp = now.replace(hour=hour, minute=minute).strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Business hours (normal)
        hour = random.randint(9, 17)
        minute = random.randint(0, 59)
        timestamp = now.replace(hour=hour, minute=minute).strftime('%Y-%m-%d %H:%M:%S')
    
    # Create transaction row
    transaction = [
        sender,
        receiver,
        amount,
        timestamp,
        1 if is_fraud else 0  # Fraud flag
    ]
    
    # Append to CSV
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(transaction)
        
        print(f"✅ Transaction added to CSV:")
        print(f"   Amount: {amount}")
        print(f"   Timestamp: {timestamp}")
        print(f"   {'🚨 FRAUDULENT PATTERN' if is_fraud else '✓ Normal transaction'}")
        return True
    except Exception as e:
        print(f"❌ Error adding to CSV: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate test blockchain transactions for fraud detection")
    parser.add_argument('--count', type=int, default=1, help='Number of transactions to generate')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between transactions in seconds (0 for no delay)')
    parser.add_argument('--fraud-rate', type=float, default=0.3, help='Probability of fraud (0.0 to 1.0, default: 0.3)')
    
    args = parser.parse_args()
    
    # Set the fraud rate
    global fraud_rate
    fraud_rate = args.fraud_rate
    
    print(f"Generating {args.count} blockchain transactions with {args.fraud_rate*100:.1f}% chance of fraud")
    
    # Track fraud statistics
    fraud_count = 0
    normal_count = 0
    
    for i in range(args.count):
        if i > 0 and args.delay > 0:
            print(f"Waiting {args.delay} seconds before next transaction...")
            time.sleep(args.delay)
            
        print(f"Creating blockchain transaction {i+1}/{args.count}")
        _, is_fraud = create_blockchain_transaction()
        
        # Update statistics
        if is_fraud:
            fraud_count += 1
        else:
            normal_count += 1
    
    print(f"Completed generating {args.count} transactions.")
    print(f"Statistics: {fraud_count} fraudulent, {normal_count} normal")
    print(f"Actual fraud rate: {fraud_count/args.count*100:.1f}%")
    print("Run 'python app.py --mode monitor' to detect these transactions.")

if __name__ == "__main__":
    main()