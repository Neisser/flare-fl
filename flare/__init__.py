"""
Flare: Federated Learning with Blockchain and IoT Focus
"""

__version__ = "0.0.1"

from flare.blockchain import (
    BlockchainConnector,
    ConsensusMechanism,
    MockChainConnector,
    MockPoAConsensus,
    TransactionPayload,
    TransactionReceipt,
)
from flare.consensus import VRFConsensus

from .compression import (
    BytesLike,
    Compressor,
    GzipCompressor,
    NoCompression,
    PowerSGDCompressor,
    ZlibCompressor,
)
from .core import FlareConfig, FlareNode, RoundContext
from .federation import (
    AggregationStrategy,
    Client,
    FedAvg,
    FederatedClient,
    MIAggregationStrategy,
    Orchestrator,
)
from .models import (
    EvalData,
    Metrics,
    MockModelAdapter,
    ModelAdapter,
    ModelWeights,
    TrainData,
)
from .storage import (
    InMemoryStorageProvider,
    StorageData,
    StorageIdentifier,
    StorageProvider,
)

print("Initializing Flare library...")  # Temp, just to show the import is working

__all__ = [
    "BlockchainConnector",
    "ConsensusMechanism",
    "MockChainConnector",
    "MockPoAConsensus",
    "TransactionPayload",
    "TransactionReceipt",
    "BytesLike",
    "Compressor",
    "GzipCompressor",
    "NoCompression",
    "PowerSGDCompressor",
    "ZlibCompressor",
    "FlareConfig",
    "FlareNode",
    "RoundContext",
    "AggregationStrategy",
    "FedAvg",
    "Client",
    "FederatedClient",
    "Orchestrator",
    "EvalData",
    "Metrics",
    "MockModelAdapter",
    "ModelAdapter",
    "ModelWeights",
    "TrainData",
    "InMemoryStorageProvider",
    "StorageData",
    "StorageIdentifier",
    "StorageProvider",
    "MIAggregationStrategy",
    "VRFConsensus",
]
