"""
OrchestratorBuilder - Fluent API for configuring Orchestrators
"""

from typing import Optional, Union

from flare.core import FlareConfig
from flare.federation import Orchestrator
from flare.federation.mi_orchestrator import MIOrchestrator
from flare.federation.vrf_orchestrator import VRFOrchestrator


class OrchestratorBuilder:
    """
    Builder pattern for creating Orchestrator instances with fluent configuration.

    Example:
        orchestrator = (
            OrchestratorBuilder()
            .with_model_adapter(PyTorchModelAdapter(model))
            .with_compressor(PowerSGDCompressor(rank=4))
            .with_storage_provider(InMemoryStorageProvider())
            .with_blockchain(MockChainConnector(), VRFConsensus())
            .with_aggregation_strategy(FedAvg())
            .with_rounds(num_rounds=3, clients_per_round=3)
            .build()
        )
    """

    def __init__(self):
        """Initialize the builder with empty configuration."""
        self._config = FlareConfig()
        self._orchestrator_type = "basic"  # basic, mi, vrf
        self._blockchain_connector = None
        self._consensus_mechanism = None

    def with_model_adapter(self, model_adapter) -> "OrchestratorBuilder":
        """Configure the model adapter for the orchestrator."""
        self._config.set("model_adapter", model_adapter)
        return self

    def with_compressor(self, compressor) -> "OrchestratorBuilder":
        """Configure the compression strategy."""
        self._config.set("compressor", compressor)
        return self

    def with_storage_provider(self, storage_provider) -> "OrchestratorBuilder":
        """Configure the storage provider."""
        self._config.set("storage_provider", storage_provider)
        return self

    def with_blockchain(
        self, connector, consensus_mechanism=None
    ) -> "OrchestratorBuilder":
        """Configure blockchain connector and consensus mechanism."""
        self._blockchain_connector = connector
        self._consensus_mechanism = consensus_mechanism
        self._config.set("blockchain_connector", connector)
        if consensus_mechanism:
            self._config.set("consensus_mechanism", consensus_mechanism)
        return self

    def with_aggregation_strategy(self, strategy) -> "OrchestratorBuilder":
        """Configure the aggregation strategy."""
        self._config.set("aggregation_strategy", strategy)
        return self

    def with_rounds(
        self, num_rounds: int, clients_per_round: int
    ) -> "OrchestratorBuilder":
        """Configure the number of rounds and clients per round."""
        self._config.set("num_rounds", num_rounds)
        self._config.set("clients_per_round", clients_per_round)
        return self

    def with_client_training_params(
        self, epochs: int, learning_rate: float
    ) -> "OrchestratorBuilder":
        """Configure client training parameters."""
        self._config.set("training_epochs", epochs)
        self._config.set("learning_rate", learning_rate)
        return self

    def with_eval_data(self, eval_data) -> "OrchestratorBuilder":
        """Configure evaluation data for the orchestrator."""
        self._config.set("eval_data", eval_data)
        return self

    def with_orchestrator_type(self, orchestrator_type: str) -> "OrchestratorBuilder":
        """
        Configure the type of orchestrator to build.

        Args:
            orchestrator_type: "basic", "mi", or "vrf"
        """
        if orchestrator_type not in ["basic", "mi", "vrf"]:
            raise ValueError("orchestrator_type must be 'basic', 'mi', or 'vrf'")
        self._orchestrator_type = orchestrator_type
        return self

    def with_mi_settings(
        self, mi_threshold: float = 0.1, min_clients: int = 2, test_data_size: int = 100
    ) -> "OrchestratorBuilder":
        """Configure MI aggregation settings (for MI orchestrator)."""
        self._config.set("mi_threshold", mi_threshold)
        self._config.set("min_clients", min_clients)
        self._config.set("test_data_size", test_data_size)
        self._orchestrator_type = "mi"
        return self

    def with_vrf_settings(
        self,
        committee_size: int = 5,
        min_committee_threshold: float = 0.6,
        vrf_seed: Optional[str] = None,
    ) -> "OrchestratorBuilder":
        """Configure VRF consensus settings (for VRF orchestrator)."""
        self._config.set("committee_size", committee_size)
        self._config.set("min_committee_threshold", min_committee_threshold)
        if vrf_seed:
            self._config.set("vrf_seed", vrf_seed)
        self._orchestrator_type = "vrf"
        return self

    def build(self) -> Union[Orchestrator, MIOrchestrator, VRFOrchestrator]:
        """
        Build the orchestrator instance with the configured settings.

        Returns:
            Configured orchestrator instance

        Raises:
            ValueError: If required configuration is missing
        """
        # Validate required configuration
        self._validate_config()

        # Build the appropriate orchestrator type
        if self._orchestrator_type == "mi":
            return self._build_mi_orchestrator()
        elif self._orchestrator_type == "vrf":
            return self._build_vrf_orchestrator()
        else:
            return self._build_basic_orchestrator()

    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        required_keys = ["model_adapter", "storage_provider"]

        for key in required_keys:
            if not self._config.get(key):
                raise ValueError(f"Missing required configuration: {key}")

        # Set defaults for optional components
        if not self._config.get("compressor"):
            from flare.compression import NoCompression

            self._config.set("compressor", NoCompression())

        if not self._config.get("aggregation_strategy"):
            from flare.federation.aggregation_strategies import FedAvg

            self._config.set("aggregation_strategy", FedAvg())

        if not self._config.get("blockchain_connector"):
            from flare.blockchain import MockChainConnector, MockPoAConsensus

            self._config.set("blockchain_connector", MockChainConnector())
            if not self._config.get("consensus_mechanism"):
                self._config.set("consensus_mechanism", MockPoAConsensus(self._config))
        elif not self._config.get("consensus_mechanism"):
            # If blockchain_connector exists but consensus_mechanism doesn't
            from flare.blockchain import MockPoAConsensus

            self._config.set("consensus_mechanism", MockPoAConsensus(self._config))

        # Set default round settings
        if not self._config.get("num_rounds"):
            self._config.set("num_rounds", 1)
        if not self._config.get("clients_per_round"):
            self._config.set("clients_per_round", 1)

    def _build_basic_orchestrator(self) -> Orchestrator:
        """Build a basic orchestrator."""
        return Orchestrator(self._config)

    def _build_mi_orchestrator(self) -> MIOrchestrator:
        """Build an MI orchestrator with robust aggregation."""
        # Ensure MI aggregation strategy is configured
        if not isinstance(
            self._config.get("aggregation_strategy"), type(None)
        ):  # Check if it's MI strategy
            # If not already MI strategy, wrap it or replace it
            from flare.federation.aggregation_strategies import MIAggregationStrategy

            mi_threshold = self._config.get("mi_threshold", 0.1)
            min_clients = self._config.get("min_clients", 2)
            test_data_size = self._config.get("test_data_size", 100)

            mi_strategy = MIAggregationStrategy(
                mi_threshold=mi_threshold,
                min_clients=min_clients,
                test_data_size=test_data_size,
            )
            self._config.set("aggregation_strategy", mi_strategy)

        return MIOrchestrator(self._config)

    def _build_vrf_orchestrator(self) -> VRFOrchestrator:
        """Build a VRF orchestrator with consensus validation."""
        # Ensure VRF consensus is configured
        if not self._config.get("consensus_mechanism"):
            from flare.consensus import VRFConsensus

            committee_size = self._config.get("committee_size", 5)
            min_committee_threshold = self._config.get("min_committee_threshold", 0.6)
            vrf_seed = self._config.get("vrf_seed")

            vrf_consensus = VRFConsensus(
                committee_size=committee_size,
                min_committee_threshold=min_committee_threshold,
                vrf_seed=vrf_seed,
            )
            self._config.set("consensus_mechanism", vrf_consensus)

        return VRFOrchestrator(self._config)

    def get_config(self) -> FlareConfig:
        """Get the current configuration (for debugging/inspection)."""
        return self._config.copy()
