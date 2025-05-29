from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

TransactionPayload = Dict[str, Any]
TransactionReceipt = Dict[str, Any]
ContractAddress = str
FunctionCallResult = Any


class BlockchainConnector(ABC):
    """Abstract base class for blockchain connectors."""

    @abstractmethod
    def submit_transaction(
        self,
        payload: TransactionPayload,
        wait_for_confirmation: bool = True
    ) -> Optional[TransactionReceipt]:
        """Submits a transaction to the blockchain."""
        pass

    @abstractmethod
    def read_state(
        self,
        contract_address: ContractAddress,
        function_name: str,
        *args
    ) -> FunctionCallResult:
        """Reads data from a smart contract state."""
        pass

    @abstractmethod
    def deploy_contract(
        self,
        contract_bytecode: Any,
        *constructor_args
    ) -> Optional[ContractAddress]:
        """Deploys a smart contract."""
        pass
