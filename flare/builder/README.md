# Flare Builder Pattern

El módulo **Builder** de Flare proporciona una API fluida y fácil de usar para configurar `Orchestrators` y `Clients` de manera modular y extensible.

## 🎯 Objetivo

Simplificar la configuración de componentes Flare eliminando la complejidad de inicializar manualmente cada componente y sus dependencias.

## 🚀 Uso Básico

### OrchestratorBuilder

```python
from flare import OrchestratorBuilder
from flare.models.pytorch_adapter import PyTorchModelAdapter
from flare.compression import PowerSGDCompressor
from flare import InMemoryStorageProvider, MockChainConnector, FedAvg

# Configuración básica
orchestrator = (
    OrchestratorBuilder()
    .with_model_adapter(PyTorchModelAdapter(model))
    .with_compressor(PowerSGDCompressor(rank=4))
    .with_storage_provider(InMemoryStorageProvider())
    .with_blockchain(MockChainConnector())
    .with_aggregation_strategy(FedAvg())
    .with_rounds(num_rounds=3, clients_per_round=5)
    .build()
)
```

### ClientBuilder

```python
from flare import ClientBuilder

client = (
    ClientBuilder()
    .with_id("client_1")
    .with_local_data((X_train, y_train))
    .with_model_adapter(PyTorchModelAdapter(model))
    .with_compressor(PowerSGDCompressor(rank=4))
    .with_storage_provider(storage_provider)
    .with_blockchain_connector(blockchain_connector)
    .as_federated_client()
    .build()
)
```

## 🔧 Características Avanzadas

### Tipos de Orquestadores

#### 1. Orquestador Básico (por defecto)
```python
orchestrator = (
    OrchestratorBuilder()
    .with_model_adapter(adapter)
    .with_storage_provider(storage)
    .build()  # Usa defaults para todo lo demás
)
```

#### 2. Orquestador MI (Robust Aggregation)
```python
orchestrator = (
    OrchestratorBuilder()
    .with_model_adapter(adapter)
    .with_storage_provider(storage)
    .with_mi_settings(
        mi_threshold=0.1,
        min_clients=2,
        test_data_size=100
    )
    .build()  # Automáticamente crea MIOrchestrator
)
```

#### 3. Orquestador VRF (Consensus Validation)
```python
orchestrator = (
    OrchestratorBuilder()
    .with_model_adapter(adapter)
    .with_storage_provider(storage)
    .with_vrf_settings(
        committee_size=5,
        min_committee_threshold=0.6,
        vrf_seed="production_seed"
    )
    .build()  # Automáticamente crea VRFOrchestrator
)
```

### Tipos de Clientes

#### Cliente Federado (por defecto)
```python
client = (
    ClientBuilder()
    .with_id("client_1")
    .with_local_data(data)
    .with_model_adapter(adapter)
    .as_federated_client()  # Soporta compresión
    .build()
)
```

#### Cliente Básico
```python
client = (
    ClientBuilder()
    .with_id("client_1")
    .with_local_data(data)
    .with_model_adapter(adapter)
    .as_basic_client()  # Sin compresión
    .build()
)
```

## ⚙️ Configuraciones por Defecto

### OrchestratorBuilder
- **Compressor**: `NoCompression()` si no se especifica
- **Aggregation Strategy**: `FedAvg()` si no se especifica
- **Blockchain**: `MockChainConnector()` + `MockPoAConsensus()` si no se especifica
- **Rounds**: `num_rounds=1, clients_per_round=1` si no se especifica

### ClientBuilder
- **Compressor**: `NoCompression()` para clientes federados si no se especifica
- **Blockchain Connector**: `MockChainConnector()` si no se especifica
- **Training Params**: `epochs=1, learning_rate=0.01` si no se especifica

## 🔍 Validación y Errores

Los builders validan automáticamente la configuración:

```python
# ❌ Error: Falta configuración requerida
orchestrator = OrchestratorBuilder().build()
# ValueError: Missing required configuration: model_adapter

# ❌ Error: Falta ID del cliente
client = ClientBuilder().with_local_data(data).build()
# ValueError: Client ID is required. Use .with_id(client_id)
```

## 📋 API Completa

### OrchestratorBuilder

| Método | Descripción |
|--------|-------------|
| `.with_model_adapter(adapter)` | Configura el adaptador de modelo |
| `.with_compressor(compressor)` | Configura la estrategia de compresión |
| `.with_storage_provider(provider)` | Configura el proveedor de almacenamiento |
| `.with_blockchain(connector, consensus=None)` | Configura blockchain y consenso |
| `.with_aggregation_strategy(strategy)` | Configura la estrategia de agregación |
| `.with_rounds(num_rounds, clients_per_round)` | Configura parámetros de rondas |
| `.with_client_training_params(epochs, lr)` | Configura parámetros de entrenamiento |
| `.with_eval_data(data)` | Configura datos de evaluación |
| `.with_orchestrator_type(type)` | Especifica tipo: "basic", "mi", "vrf" |
| `.with_mi_settings(threshold, min_clients, test_size)` | Configuración MI |
| `.with_vrf_settings(committee_size, threshold, seed)` | Configuración VRF |
| `.build()` | Construye el orquestador |

### ClientBuilder

| Método | Descripción |
|--------|-------------|
| `.with_id(client_id)` | **Requerido**: ID del cliente |
| `.with_local_data(data)` | **Requerido**: Datos locales |
| `.with_model_adapter(adapter)` | **Requerido**: Adaptador de modelo |
| `.with_compressor(compressor)` | Configura compresión |
| `.with_storage_provider(provider)` | **Requerido**: Proveedor de almacenamiento |
| `.with_blockchain_connector(connector)` | Configura conector blockchain |
| `.with_consensus(mechanism)` | Configura mecanismo de consenso |
| `.with_training_params(epochs, lr, batch_size)` | Parámetros de entrenamiento |
| `.with_device(device)` | Dispositivo para entrenamiento |
| `.as_federated_client()` | Construir como FederatedClient |
| `.as_basic_client()` | Construir como Client básico |
| `.build()` | Construye el cliente |

## 🌟 Beneficios

1. **API Fluida**: Encadenamiento de métodos legible
2. **Defaults Inteligentes**: Componentes opcionales con valores por defecto
3. **Validación Automática**: Errores claros para configuración faltante
4. **Extensibilidad**: Fácil agregar nuevos tipos y configuraciones
5. **Menos Código**: Reduce significativamente el boilerplate
6. **Type Safety**: Mejor experiencia de desarrollo con hints de tipo

## 📁 Ejemplos

Consulta `examples/builder_example.py` para ejemplos completos de uso del patrón Builder. 