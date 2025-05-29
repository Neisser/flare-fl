# flare/blockchain/consensus.py
from abc import ABC, abstractmethod
from typing import Any, List

from flare.core.base_classes import FlareConfig

# from .connectors import BlockchainConnector # Descomentar si es necesario


class ConsensusMechanism(ABC):
    """
    Abstract base class for consensus mechanisms over federated learning artifacts
    (e.g., model updates, global model validity).
    This is distinct from the underlying blockchain's consensus (PoW, PoS).
    """
    # Use 'FlareConfig' to avoid circular import issues
    def __init__(self, config: 'FlareConfig'):
        self.config = config
        # TODO: Review if we need a blockchain connector here
        # self.blockchain_connector = config.get_required('blockchain_connector') # Si interactúa directamente

    @abstractmethod
    def validate_contribution(self, contribution: Any, **kwargs) -> bool:
        """Validates an individual contribution (e.g., model update)."""
        pass

    @abstractmethod
    def reach_agreement(self, proposals: List[Any], **kwargs) -> Any:
        """
        Processes a list of proposals (e.g., aggregated models, update quality scores)
        and returns the agreed-upon result or decision.
        """
        pass
