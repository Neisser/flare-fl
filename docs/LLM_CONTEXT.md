# LLM Context for Flare Project

This document provides detailed architectural and class-level information for the Flare project,
to be used by the Large Language Model (LLM) for contextualizing its responses.

---

## 1. Project Overview

**Project Name:** Flare
**Core Purpose:** A Python library for simulating and implementing federated learning (FL) systems
that leverage blockchain for coordination, trust, and logging, with a strong emphasis on
efficiency and applicability for IoT devices.

**Key Design Principles:**
- **Modularity:** Components are interchangeable via abstract interfaces.
- **Abstraction:** Clear interfaces for developers to implement custom components.
- **Simplicity (User-facing):** Complex backend, simple API.
- **Efficiency:** Optimized for IoT (communication, computation).
- **Security & Privacy:** Crucial for FL and blockchain (future focus).

---

## 2. Directory Structure

```
flare-library/
├── flare/                      # Main library package
│   ├── __init__.py
│   ├── core/                   # Core utilities and base classes
│   │   ├── __init__.py
│   │   ├── base_classes.py     # FlareConfig, FlareNode, RoundContext
│   │   └── utils.py            # Generic core utilities
│   ├── models/                 # Model management and adapters
│   │   ├── __init__.py
│   │   └── adapters.py         # ModelAdapter, MockModelAdapter (and future PyTorch, TF, SKLearn)
│   ├── compression/            # Model update compression strategies
│   │   ├── __init__.py
│   │   └── compressors.py      # Compressor, NoCompression, ZlibCompressor, GzipCompressor
│   ├── blockchain/             # Blockchain interface and consensus mechanisms
│   │   ├── __init__.py
│   │   ├── connectors.py       # BlockchainConnector, MockChainConnector
│   │   └── consensus.py        # ConsensusMechanism, MockPoAConsensus
│   ├── storage/                # Data storage providers
│   │   ├── __init__.py
│   │   └── providers.py        # StorageProvider, InMemoryStorageProvider (and future IPFS)
│   ├── federation/             # Federated learning core logic
│   │   ├── __init__.py
│   │   ├── client.py           # Client class
│   │   ├── orchestrator.py     # Orchestrator class
│   │   └── strategies.py       # AggregationStrategy, FedAvg
│   └── utils/                  # General library utilities
│       ├── __init__.py
│       ├── serialization.py
│       └── data_loader.py      # For example dataset loading/partitioning
│
├── examples/                   # Usage examples
│   ├── __init__.py
│   └── simple_mnist_simulation.py
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   └── core/
│       └── test_config.py
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 3. Key Classes and Interfaces (with primary methods)

### `flare.core`
- **`FlareConfig`**:
    - `set(key, value)`: Set a configuration value.
    - `get(key, default=None)`: Get a configuration value.
    - `get_required(key)`: Get a required configuration value (raises ValueError if not found).
    - `copy()`: Return a deep copy of the config.
- **`FlareNode` (ABC)**:
    - `__init__(node_id, config)`
    - `start()`: Abstract method to start node operations.
    - `stop()`: Abstract method to stop node operations.
- **`RoundContext`**:
    - `round_number`: Current round number.
    - `global_model_version`: Identifier for the global model version.
    - `metadata`: Dictionary for additional round-specific data.

### `flare.models`
- **`ModelAdapter` (ABC)**:
    - `get_weights() -> ModelWeights`: Get current model weights.
    - `set_weights(weights: ModelWeights)`: Set model weights.
    - `train(data: TrainData, epochs, learning_rate, **kwargs) -> Dict[str, Any]`: Train locally.
    - `evaluate(data: EvalData, **kwargs) -> Metrics`: Evaluate model.
    - `predict(data: Any, **kwargs) -> Any`: Make predictions.
    - `serialize_model() -> bytes`: Serialize full model (architecture + weights).
    - `deserialize_model(model_bytes: bytes)`: Deserialize full model.
    - `serialize_weights() -> bytes`: Serialize only weights.
    - `deserialize_weights(weights_bytes: bytes) -> ModelWeights`: Deserialize weights.
- **`MockModelAdapter`**: Basic mock implementation of `ModelAdapter`.
- **Type Aliases**: `ModelWeights`, `ModelInstance`, `TrainData`, `EvalData`, `Metrics`.

### `flare.compression`
- **`Compressor` (ABC)**:
    - `compress(data: BytesLike) -> BytesLike`: Compress data.
    - `decompress(data: BytesLike) -> BytesLike`: Decompress data.
- **`NoCompression`**: Passthrough compressor.
- **`ZlibCompressor`**: Zlib compression.
- **`GzipCompressor`**: Gzip compression.
- **Type Alias**: `BytesLike`.

### `flare.storage`
- **`StorageProvider` (ABC)**:
    - `put(identifier, data) -> Optional[StorageIdentifier]`: Store data.
    - `get(identifier) -> Optional[StorageData]`: Retrieve data.
    - `delete(identifier) -> bool`: Delete data.
    - `exists(identifier) -> bool`: Check existence.
- **`InMemoryStorageProvider`**: In-memory dictionary storage.
- **Type Aliases**: `StorageIdentifier`, `StorageData`.

### `flare.blockchain`
- **`BlockchainConnector` (ABC)**:
    - `submit_transaction(payload: TransactionPayload, wait_for_confirmation=True) -> Optional[TransactionReceipt]`: Submit transaction.
    - `read_state(contract_address, function_name, *args) -> FunctionCallResult`: Read contract state.
- **`MockChainConnector`**: In-memory mock blockchain.
- **`ConsensusMechanism` (ABC)**:
    - `validate_contribution(contribution: Any, **kwargs) -> bool`: Validate single contribution.
    - `reach_agreement(proposals: List[Any], **kwargs) -> Any`: Reach agreement on proposals.
- **`MockPoAConsensus`**: Mock Proof-of-Authority consensus.
- **Type Aliases**: `TransactionPayload`, `TransactionReceipt`, `ContractAddress`, `FunctionCallResult`.

### `flare.federation`
- **`Client` (inherits `FlareNode`)**:
    - `__init__(client_id, local_data, config)`
    - `receive_global_model(model_ref: StorageIdentifier)`: Fetch and load global model.
    - `train_local(round_context, epochs, learning_rate)`: Perform local training.
    - `send_update(round_context, updated_weights)`: Compress, store, and log update.
    - `evaluate_local_model(eval_data)`: Evaluate local model.
- **`Orchestrator` (inherits `FlareNode`)**:
    - `__init__(config)`
    - `register_client(client)`: Register a client.
    - `select_clients() -> List[Client]`: Select clients for a round.
    - `execute_round()`: Main method to run a FL round.
    - `evaluate_global_model(eval_data)`: Evaluate current global model.
- **`AggregationStrategy` (ABC)**:
    - `aggregate(local_model_updates, client_data_sizes=None, previous_global_weights=None) -> ModelWeights`: Aggregate updates.
- **`FedAvg`**: Federated Averaging strategy.

---

## 4. Current Project State

- **Boilerplate:** Directory structure and initial empty `__init__.py` files are set up.
- **Core Interfaces:** `FlareConfig`, `FlareNode`, `RoundContext` are defined.
- **Module Interfaces & Mocks:**
    - `flare.models`: `ModelAdapter` interface and `MockModelAdapter` implemented.
    - `flare.compression`: `Compressor` interface and `NoCompression`, `ZlibCompressor`, `GzipCompressor` implemented.
    - `flare.storage`: `StorageProvider` interface and `InMemoryStorageProvider` implemented.
    - `flare.blockchain`: `BlockchainConnector` interface (`MockChainConnector`) and `ConsensusMechanism` interface (`MockPoAConsensus`) implemented.
    - `flare.federation`: `AggregationStrategy` interface (`FedAvg`) implemented.
- **FL Logic:** `Client` and `Orchestrator` classes are implemented, utilizing the mock components to demonstrate a basic FL round flow.
- **Simulation:** `examples/simple_mnist_simulation.py` is set up to run a basic simulation using all the mock components.
- **Testing:** Basic `pytest` setup with a test for `FlareConfig`.

---

## 5. Next Steps (as of last interaction)

The immediate next steps involve:
- Implementing concrete `ModelAdapter` for PyTorch/TensorFlow.
- Expanding `StorageProvider` with `IPFSStorageProvider`.
- Expanding `BlockchainConnector` with `Web3Connector` for real Ethereum interaction.
- Adding more sophisticated `ConsensusMechanism` implementations.
- Adding more `AggregationStrategy` implementations (e.g., FedProx).
- Enhancing the simulation environment with more realistic data loading and metrics.