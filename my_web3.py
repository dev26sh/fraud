from web3 import Web3

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))

# Verify connection
print(f"Connected: {w3.is_connected()}")
print(f"Chain ID: {w3.eth.chain_id}")
print(f"Block number: {w3.eth.block_number}")