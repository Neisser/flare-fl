"""
ClientBuilder - Fluent API for configuring Federated Clients
"""

from typing import Any, Optional, Union

from flare.core import FlareConfig
from flare.federation import Client, FederatedClient


class ClientBuilder:
    """
    Builder pattern for creating Client instances with fluent configuration.

    Example:
        client = (
            ClientBuilder()
            .with_id("client_1")
            .with_local_data((X_train, y_train))
            .with_model_adapter(PyTorchModelAdapter(model))
            .with_compressor(PowerSGDCompressor(rank=4))
            .with_storage_provider(storage_provider)
            .with_blockchain_connector(blockchain_connector)
            .build()
        )
    """

    def __init__(self):
        """Initialize the builder with empty configuration."""
        self._config = FlareConfig()
        self._client_id = None
        self._local_data = None
        self._client_type = "federated"  # "basic" or "federated"

    def with_id(self, client_id: str) -> "ClientBuilder":
        """Configure the client identifier."""
        self._client_id = client_id
        return self

    def with_local_data(self, local_data: Any) -> "ClientBuilder":
        """Configure the local training data for the client."""
        self._local_data = local_data
        return self

    def with_model_adapter(self, model_adapter) -> "ClientBuilder":
        """Configure the model adapter for the client."""
        self._config.set("model_adapter", model_adapter)
        return self

    def with_compressor(self, compressor) -> "ClientBuilder":
        """Configure the compression strategy."""
        self._config.set("compressor", compressor)
        return self

    def with_storage_provider(self, storage_provider) -> "ClientBuilder":
        """Configure the storage provider."""
        self._config.set("storage_provider", storage_provider)
        return self

    def with_blockchain_connector(self, blockchain_connector) -> "ClientBuilder":
        """Configure the blockchain connector."""
        self._config.set("blockchain_connector", blockchain_connector)
        return self

    def with_consensus(self, consensus_mechanism) -> "ClientBuilder":
        """Configure the consensus mechanism."""
        self._config.set("consensus_mechanism", consensus_mechanism)
        return self

    def with_training_params(
        self,
        epochs: int = 1,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
    ) -> "ClientBuilder":
        """Configure training parameters."""
        self._config.set("training_epochs", epochs)
        self._config.set("learning_rate", learning_rate)
        if batch_size is not None:
            self._config.set("batch_size", batch_size)
        return self

    def with_client_type(self, client_type: str) -> "ClientBuilder":
        """
        Configure the type of client to build.

        Args:
            client_type: "basic" or "federated"
        """
        if client_type not in ["basic", "federated"]:
            raise ValueError("client_type must be 'basic' or 'federated'")
        self._client_type = client_type
        return self

    def as_federated_client(self) -> "ClientBuilder":
        """Configure to build a FederatedClient (with compression support)."""
        self._client_type = "federated"
        return self

    def as_basic_client(self) -> "ClientBuilder":
        """Configure to build a basic Client."""
        self._client_type = "basic"
        return self

    def with_device(self, device: str) -> "ClientBuilder":
        """Configure the device for training (e.g., 'cpu', 'cuda')."""
        self._config.set("device", device)
        return self

    def build(self) -> Union[Client, FederatedClient]:
        """
        Build the client instance with the configured settings.

        Returns:
            Configured client instance

        Raises:
            ValueError: If required configuration is missing
        """
        # Validate required configuration
        self._validate_config()

        # Build the appropriate client type
        if self._client_type == "federated":
            return self._build_federated_client()
        else:
            return self._build_basic_client()

    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        # Validate required parameters
        if not self._client_id:
            raise ValueError("Client ID is required. Use .with_id(client_id)")

        if self._local_data is None:
            raise ValueError("Local data is required. Use .with_local_data(data)")

        required_keys = ["model_adapter", "storage_provider"]
        for key in required_keys:
            if not self._config.get(key):
                raise ValueError(f"Missing required configuration: {key}")

        # Set defaults for optional components
        if not self._config.get("compressor") and self._client_type == "federated":
            from flare.compression import NoCompression

            self._config.set("compressor", NoCompression())

        if not self._config.get("blockchain_connector"):
            from flare.blockchain import MockChainConnector

            self._config.set("blockchain_connector", MockChainConnector())

        # Set default training parameters
        if not self._config.get("training_epochs"):
            self._config.set("training_epochs", 1)
        if not self._config.get("learning_rate"):
            self._config.set("learning_rate", 0.01)

    def _build_basic_client(self) -> Client:
        """Build a basic client."""
        if self._client_id is None:
            raise ValueError("Client ID must be set before building")
        return Client(
            client_id=self._client_id, local_data=self._local_data, config=self._config
        )

    def _build_federated_client(self) -> FederatedClient:
        """Build a federated client with compression support."""
        if self._client_id is None:
            raise ValueError("Client ID must be set before building")
        return FederatedClient(
            client_id=self._client_id, local_data=self._local_data, config=self._config
        )

    def get_config(self) -> FlareConfig:
        """Get the current configuration (for debugging/inspection)."""
        return self._config.copy()

    def get_client_id(self) -> Optional[str]:
        """Get the configured client ID."""
        return self._client_id

    def get_local_data(self) -> Any:
        """Get the configured local data."""
        return self._local_data
