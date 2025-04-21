// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {
    struct Transaction {
        uint256 id;
        address sender;
        address receiver;
        uint256 amount;
        uint256 timestamp;
        bool isFlagged;
        string mlConfidence;
    }
    
    struct FraudReport {
        uint256 transactionId;
        address reporter;
        string reason;
        uint256 timestamp;
        bool resolved;
    }
    
    mapping(uint256 => Transaction) public transactions;
    mapping(uint256 => FraudReport[]) public fraudReports;
    
    uint256 public transactionCount = 0;
    
    // Events
    event TransactionAdded(uint256 indexed id, address sender, address receiver, uint256 amount);
    event TransactionFlagged(uint256 indexed id, string confidence);
    event FraudReported(uint256 indexed transactionId, address reporter, string reason);
    
    // Add a new transaction
    function addTransaction(address _receiver, uint256 _amount) public returns (uint256) {
        transactionCount++;
        
        transactions[transactionCount] = Transaction({
            id: transactionCount,
            sender: msg.sender,
            receiver: _receiver,
            amount: _amount,
            timestamp: block.timestamp,
            isFlagged: false,
            mlConfidence: "0.0"
        });
        
        emit TransactionAdded(transactionCount, msg.sender, _receiver, _amount);
        
        return transactionCount;
    }
    
    // Flag a transaction     pip install py-solc-xas potentially fraudulent (called by authorized ML oracles)
    function flagTransaction(uint256 _id, string memory _confidence) public {
        require(_id <= transactionCount, "Transaction does not exist");
        
        transactions[_id].isFlagged = true;
        transactions[_id].mlConfidence = _confidence;
        
        emit TransactionFlagged(_id, _confidence);
    }
    
    // Report fraud for a transaction
    function reportFraud(uint256 _transactionId, string memory _reason) public {
        require(_transactionId <= transactionCount, "Transaction does not exist");
        
        FraudReport memory report = FraudReport({
            transactionId: _transactionId,
            reporter: msg.sender,
            reason: _reason,
            timestamp: block.timestamp,
            resolved: false
        });
        
        fraudReports[_transactionId].push(report);
        
        emit FraudReported(_transactionId, msg.sender, _reason);
    }
    
    // Get transaction details
    function getTransaction(uint256 _id) public view returns (
        uint256, address, address, uint256, uint256, bool, string memory
    ) {
        require(_id <= transactionCount, "Transaction does not exist");
        
        Transaction memory txn = transactions[_id];
        
        return (
            txn.id,
            txn.sender,
            txn.receiver,
            txn.amount,
            txn.timestamp,
            txn.isFlagged,
            txn.mlConfidence
        );
    }
    
    // Get fraud reports count for a transaction
    function getFraudReportsCount(uint256 _transactionId) public view returns (uint256) {
        return fraudReports[_transactionId].length;
    }
}