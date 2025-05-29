from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class FlareConfig:
    """
    A simple configuration class for Flare components.
    Uses a dictionary-like interface.
    """

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = initial_config if initial_config else {}

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._config.get(key, default)

    def get_required(self, key: str) -> Any:
        if key not in self._config:
            raise ValueError(f"Required configuration key '{key}' not found.")
        return self._config[key]

    def all(self) -> Dict[str, Any]:
        return self._config.copy()

    def copy(self) -> "FlareConfig":
        return FlareConfig(self._config.copy())

    def __str__(self) -> str:
        return str(self._config)


class FlareNode(ABC):
    """
    Abstract base class for any participating node in the Flare network
    (e.g., Client, Orchestrator).
    """

    def __init__(self, node_id: str, config: FlareConfig):
        self.node_id = node_id
        self.config = config

    @abstractmethod
    def start(self):
        """Start the node's main operation or event loop."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the node and clean up resources."""
        pass


class RoundContext:
    """
    Carries context information for a specific federated learning round.
    """

    def __init__(
        self,
        round_number: int,
        global_model_version: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.round_number = round_number
        self.global_model_version = (
            global_model_version  # Could be a hash, CID, or version number
        )
        self.metadata: Dict[str, Any] = (
            metadata if metadata is not None else {}
        )  # For any other round-specific data

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any:
        return self.metadata.get(key, default)

    def __str__(self) -> str:
        return (
            f"RoundContext(round={self.round_number}, "
            f"model_version={self.global_model_version}, "
            f"metadata={self.metadata})"
        )
