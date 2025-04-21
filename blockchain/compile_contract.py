from solcx import compile_standard, install_solc
import json
import os

def compile_contract():
    print("Installing solc...")
    # Install specific solc version
    install_solc("0.8.0")
    
    print("Reading contract source...")
    # Read the Solidity contract
    with open('blockchain/contracts/FraudDetection.sol', 'r') as file:
        contract_source = file.read()
    
    print("Compiling contract...")
    # Compile the contract
    compiled_sol = compile_standard({
        "language": "Solidity",
        "sources": {
            "FraudDetection.sol": {
                "content": contract_source
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                }
            }
        }
    }, solc_version="0.8.0")
    
    # Create the output directory if it doesn't exist
    os.makedirs('blockchain/contracts/compiled', exist_ok=True)
    
    # Extract the contract data
    contract_data = compiled_sol['contracts']['FraudDetection.sol']['FraudDetection']
    abi = contract_data['abi']
    bytecode = contract_data['evm']['bytecode']['object']
    
    # Format for deploy_contract.py
    output = {
        'abi': abi,
        'bytecode': bytecode
    }
    
    print("Saving compiled contract...")
    # Save to file
    with open('blockchain/contracts/compiled/FraudDetection.json', 'w') as file:
        json.dump(output, file)
    
    print("Contract compiled successfully")

if __name__ == '__main__':
    compile_contract()