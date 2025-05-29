from typing import Any, Dict, Optional, Union

from .connectors import (
    BlockchainConnector,
    ContractAddress,
    FunctionCallResult,
    TransactionPayload,
    TransactionReceipt,
)


class MockChainConnector(BlockchainConnector):
    """
    A mock blockchain connector for simulation without a real blockchain.
    Simulates transactions and state in memory.
    """
    def __init__(self):
        # Transaction logs
        self._chain_log: list[Dict[str, Any]] = []
        
        # Simulated state storage for simple contract
        self._state: Dict[str, Any] = {}
        self._tx_counter = 0
        print("MockChainConnector initialized.")

    def submit_transaction(self,
                           payload: TransactionPayload,
                           wait_for_confirmation: bool = True
                           ) -> Optional[TransactionReceipt]:
        self._tx_counter += 1
        tx_hash = f"0xmocktx{self._tx_counter:06d}"
        transaction = {
            "tx_hash": tx_hash,
            "payload": payload,
            # Simulate inmediate confirmation
            "status": "success",
            # Simulate a block number
            "block_number": len(self._chain_log) + 1
        }
        self._chain_log.append(transaction)
        print(f"MockChain: Submitted transaction {tx_hash} with payload: {payload}")

        # Simulate state update based on payload
        if payload.get("action") == "update_value" and "key" in payload and "value" in payload:
            self._state[payload["key"]] = payload["value"]
            print(f"MockChain: State updated - {payload['key']} = {payload['value']}")

        return transaction if wait_for_confirmation else {"tx_hash": tx_hash, "status": "pending"}

    def read_state(self, contract_address: ContractAddress, function_name: str, *args) -> FunctionCallResult:
        # Basic simulation: contract_address could be a key in self._state
        # and function_name could be a sub-key or ignored.
        print(f"MockChain: Reading state from '{contract_address}', function '{function_name}', args: {args}")
        if contract_address == "global_model_registry":
            if function_name == "get_latest_model_hash":
                return self._state.get("latest_model_hash", None)
            elif function_name == "get_model_version_info" and args:
                return self._state.get(f"model_version_{args[0]}", None)
        # Return default value or None if not found
        # Ensure the key is a string and not None
        fallback_key = str(args[0]) if args and args[0] is not None else None
        if fallback_key is not None:
            return self._state.get(function_name, self._state.get(fallback_key, None))
        else:
            return self._state.get(function_name, None)

    # Helper para inspección
    def get_chain_log(self):
        return self._chain_log

    # Helper para inspección
    def get_current_state(self):
        return self._state

    def deploy_contract(self, contract_bytecode: Any, *constructor_args) -> Union[ContractAddress, None]:
        raise NotImplementedError
