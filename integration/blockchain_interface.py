from web3 import Web3
import json
import os

class BlockchainInterface:
    def __init__(self, contract_address=None, contract_abi=None):
        # Connect to Ethereum node
        self.w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))  # Use your Ethereum node or Infura URL
        
        # Set default account
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        
        # Load contract information
        if contract_address is None or contract_abi is None:
            contract_address, contract_abi = self._load_contract_info()
        
        # Initialize contract
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
    
    def _load_contract_info(self):
        # Load contract address
        with open('blockchain/contract_address.txt', 'r') as file:
            contract_address = file.read().strip()
        
        # Load contract ABI
        with open('blockchain/contracts/compiled/FraudDetection.json', 'r') as file:
            compiled_contract = json.load(file)
            contract_abi = compiled_contract['abi']
        
        return contract_address, contract_abi
    
    def add_transaction(self, receiver, amount):
        # Add transaction to blockchain
        tx_hash = self.contract.functions.addTransaction(receiver, amount).transact()
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get transaction ID from event logs
        logs = self.contract.events.TransactionAdded().process_receipt(receipt)
        transaction_id = logs[0]['args']['id']
        
        return transaction_id
    
    def flag_transaction(self, transaction_id, confidence):
        # Flag transaction as potentially fraudulent
        tx_hash = self.contract.functions.flagTransaction(
            transaction_id, 
            str(confidence)
        ).transact()
        
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    
    def report_fraud(self, transaction_id, reason):
        # Report fraud for a transaction
        tx_hash = self.contract.functions.reportFraud(transaction_id, reason).transact()
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    
    def get_transaction(self, transaction_id):
        # Get transaction details from blockchain
        tx_data = self.contract.functions.getTransaction(transaction_id).call()
        
        # Format transaction data
        transaction = {
            'id': tx_data[0],
            'sender': tx_data[1],
            'receiver': tx_data[2],
            'amount': tx_data[3],
            'timestamp': tx_data[4],
            'is_flagged': tx_data[5],
            'ml_confidence': tx_data[6]
        }
        
        return transaction