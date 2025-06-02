<div align="center">
  <img src="https://i.imgur.com/6vQYSqr.png">
</div>

# Flare: Federated Learning with Blockchain and IoT Focus

**Flare** es una biblioteca modular de Python para simular e implementar sistemas de **Entrenamiento Federado (FL)** que aprovechan **blockchain** para coordinación, confianza y registro, con un fuerte énfasis en la eficiencia y aplicabilidad para dispositivos **IoT**.

## 🚀 IMPLEMENTACIÓN ACTUAL

### ✅ FASE 1 - Completada

- **FederatedClient**: Cliente mejorado que computa diferencias de pesos (ΔW) y aplica compresión antes de enviar actualizaciones
- **PowerSGDCompressor**: Compresor que implementa aproximación de bajo rango usando iteración de potencia (Algoritmo 2 de BEFL)
- **PyTorchModelAdapter**: Adaptador para modelos PyTorch con serialización/deserialización
- **Simulación completa**: Ejemplo funcional que demuestra el flujo completo de FL con compresión

### ✅ FASE 2 - Completada: Agregación Robusta con MI

- **MIAggregationStrategy**: Estrategia de agregación basada en Mutual Information para detectar y filtrar clientes maliciosos
- **Detección de ataques**: Filtrado automático de contribuciones maliciosas usando análisis de MI entre salidas de modelos
- **MaliciousClient**: Cliente de prueba que simula diferentes tipos de ataques (ruido, aleatorio, opuesto)
- **Robustez demostrada**: FL resiliente ante clientes comprometidos con filtrado inteligente

### ✅ FASE 3 - Completada: VRF Consensus

- **VRFConsensus**: Mecanismo de consenso basado en Verifiable Random Function para selección de comité
- **Selección de comité**: Selección verificable y determinista de validadores usando VRF
- **Validación por comité**: Validación descentralizada de modelos agregados con votación
- **Tolerancia bizantina**: Resistencia a fallas y ataques mediante consenso distribuido

### 🎯 Algoritmo VRF Consensus (FASE 3)

La estrategia `VRFConsensus` implementa:

1. **Selección VRF**: Genera comité de validación usando función aleatoria verificable
2. **Propuesta de validación**: Crea propuestas para validar modelos agregados
3. **Votación del comité**: Miembros del comité votan independientemente
4. **Consenso umbral**: Aprueba/rechaza según umbral mínimo de acuerdo

## 📦 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Neisser/flare-fl
cd flarepy

# Instalar dependencias (incluye scikit-learn para MI)
pip install -r requirements.txt

# Instalar Flare en modo desarrollo
pip install -e .
```

## 🔧 Uso Rápido

### Builder Pattern (Recomendado) ✨

**Flare** ahora incluye un **patrón Builder** que hace la configuración mucho más limpia e intuitiva:

```python
from flare import OrchestratorBuilder, ClientBuilder
from flare.models.pytorch_adapter import PyTorchModelAdapter
from flare import PowerSGDCompressor, InMemoryStorageProvider

# Orquestador con API fluida
orchestrator = (
    OrchestratorBuilder()
    .with_model_adapter(PyTorchModelAdapter(model))
    .with_compressor(PowerSGDCompressor(rank=4))
    .with_storage_provider(InMemoryStorageProvider())
    .with_rounds(num_rounds=3, clients_per_round=5)
    .with_mi_settings(mi_threshold=0.1)  # Robust aggregation
    .build()  # Automáticamente crea MIOrchestrator
)

# Clientes con configuración simple
client = (
    ClientBuilder()
    .with_id("client_1")
    .with_local_data((X_train, y_train))
    .with_model_adapter(PyTorchModelAdapter(model))
    .with_compressor(PowerSGDCompressor(rank=4))
    .with_storage_provider(storage_provider)
    .as_federated_client()
    .build()
)
```

### Ejemplo FASE 1 - Compresión PowerSGD

```python
import torch
import torch.nn as nn
from flare import FlareConfig, FederatedClient, PowerSGDCompressor
from flare.models.pytorch_adapter import PyTorchModelAdapter

# Definir modelo simple
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Configurar cliente federado
model = SimpleMLP()
adapter = PyTorchModelAdapter(model)

config = FlareConfig()
config.set('model_adapter', adapter)
config.set('compressor', PowerSGDCompressor(rank=4, power_iterations=1))

# Crear cliente y entrenar
client = FederatedClient("client_1", (X_train, y_train), config)
delta_weights = client.train_local(round_context, epochs=3, learning_rate=0.01)
```

### Ejemplo FASE 2 - Agregación con MI

```python
from flare import MIAggregationStrategy

# Configurar agregación robusta
mi_strategy = MIAggregationStrategy(
    mi_threshold=0.1,      # Umbral de similitud MI
    min_clients=2,         # Mínimo de clientes confiables
    test_data_size=100     # Tamaño de datos para MI
)

# Agregar con filtrado de maliciosos
aggregated_weights = mi_strategy.aggregate(
    local_model_updates=client_updates,
    client_data_sizes=data_sizes,
    previous_global_weights=global_weights,
    test_data=(X_test, y_test)  # Datos para MI
)
```

### Ejemplo FASE 3 - VRF Consensus

```python
from flare import VRFConsensus

# Configurar consenso VRF
vrf_consensus = VRFConsensus(
    committee_size=5,             # Tamaño del comité
    min_committee_threshold=0.6,  # 60% de acuerdo mínimo
    vrf_seed="demo_seed"          # Semilla para reproducibilidad
)

# Seleccionar comité para validación
committee = vrf_consensus.select_committee(
    available_nodes=client_list,
    round_number=round_num
)

# Proponer validación de modelo
proposal_id = vrf_consensus.propose_decision(
    proposal_data={
        "round_number": round_num,
        "model_hash": model_hash,
        "validation_type": "aggregated_model"
    },
    proposer_id="orchestrator"
)

# Votar en el comité
for member_id in committee:
    validation_score = validate_model(aggregated_weights)
    vote = validation_score > threshold
    vrf_consensus.vote(proposal_id, member_id, vote)

# Obtener resultado del consenso
result = vrf_consensus.get_consensus_result(proposal_id)
model_approved = result["result"] == "approved"
```

### Ejecutar Simulaciones

```bash
# FASE 1: Compresión PowerSGD
python examples/simple_simulation.py

# FASE 2: Agregación con MI + Clientes maliciosos
python examples/phase2_mi_simulation.py

# FASE 3: VRF Consensus + Validación por comité
python examples/phase3_vrf_simulation.py

# Builder Pattern Demo (Nuevo)
python examples/builder_example.py

# Simulación Comparativa - Demuestra Builder Pattern (Nuevo)
python examples/simple_comparative_simulation.py
```

## 🏗️ Arquitectura Modular

### Módulos Principales

- **`flare.core`**: Clases base (`FlareConfig`, `FlareNode`, `RoundContext`)
- **`flare.models`**: Adaptadores de modelos (`ModelAdapter`, `PyTorchModelAdapter`)
- **`flare.compression`

## 🔧 Builder Pattern - API Mejorada

**Flare** incluye un patrón Builder que simplifica significativamente la configuración:

### Beneficios del Builder

- ✅ **API Fluida**: Encadenamiento de métodos legible
- ✅ **Defaults Inteligentes**: Componentes opcionales con valores por defecto
- ✅ **Validación Automática**: Errores claros para configuración faltante
- ✅ **Extensibilidad**: Fácil agregar nuevos tipos y configuraciones
- ✅ **Menos Código**: Reduce significativamente el boilerplate

### Tipos de Orquestadores Disponibles

```python
# Básico
basic_orch = OrchestratorBuilder().with_model_adapter(adapter).build()

# Con MI (Robust Aggregation)
mi_orch = OrchestratorBuilder().with_mi_settings(mi_threshold=0.1).build()

# Con VRF (Consensus Validation)
vrf_orch = OrchestratorBuilder().with_vrf_settings(committee_size=5).build()
```

Consulta `flare/builder/README.md` para documentación completa del Builder Pattern.