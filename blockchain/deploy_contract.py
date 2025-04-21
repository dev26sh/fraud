from web3 import Web3
import json
import os

def deploy_contract():
    # Connect to Ethereum node
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))  # Use your Ethereum node or Infura URL
    
    # Set default account
    w3.eth.default_account = w3.eth.accounts[0]
    
    # Get contract ABI and bytecode
    with open('blockchain/contracts/compiled/FraudDetection.json', 'r') as file:
        compiled_contract = json.load(file)
    
    abi = compiled_contract['abi']
    bytecode = compiled_contract['bytecode']
    
    # Deploy contract
    FraudDetection = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = FraudDetection.constructor().transact()
    
    # Wait for transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    contract_address = tx_receipt.contractAddress
    print(f'Contract deployed at address: {contract_address}')
    
    # Save contract address for later use
    with open('blockchain/contract_address.txt', 'w') as file:
        file.write(contract_address)
    
    return contract_address, abi

if __name__ == '__main__':
    deploy_contract()