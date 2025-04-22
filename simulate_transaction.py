from integration.blockchain_interface import BlockchainInterface
from web3 import Web3

# Initialize blockchain interface
blockchain = BlockchainInterface()

# Details for the new transaction
receiver_address = Web3.to_checksum_address('0x1234567890abcdef1234567890abcdef12345678')  # Use a valid address
amount = 100  # Amount you want to send

# Add the transaction
try:
    tx_id = blockchain.add_transaction(receiver_address, amount)
    print(f"✅ Successfully added transaction. Transaction ID: {tx_id}")
except Exception as e:
    print(f"❌ Error adding transaction: {e}")
