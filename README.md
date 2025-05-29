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
git clone <repository-url>
cd flarepy

# Instalar dependencias (incluye scikit-learn para MI)
pip install -r requirements.txt

# Instalar Flare en modo desarrollo
pip install -e .
```

## 🔧 Uso Rápido

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
```

## 🏗️ Arquitectura Modular

### Módulos Principales

- **`flare.core`**: Clases base (`FlareConfig`, `FlareNode`, `RoundContext`)
- **`flare.models`**: Adaptadores de modelos (`ModelAdapter`, `PyTorchModelAdapter`)
- **`flare.compression`**: Compresores (`PowerSGDCompressor`, `GzipCompressor`)
- **`flare.federation`**: Lógica FL (`FederatedClient`, `Orchestrator`, `MIAggregationStrategy`)
- **`flare.consensus`**: Mecanismos de consenso (`VRFConsensus`)
- **`flare.blockchain`**: Conectores blockchain (`MockChainConnector`)
- **`flare.storage`**: Proveedores de almacenamiento (`InMemoryStorageProvider`)

### Interfaces Clave - FASE 3

```python
# Consenso VRF para validación distribuida
class VRFConsensus(ConsensusMechanism):
    def select_committee(self, available_nodes, round_number, **kwargs) -> List[str]:
        # Selección determinista de comité usando VRF
        
    def propose_decision(self, proposal_data, proposer_id) -> str:
        # Crear propuesta para validación del comité
        
    def vote(self, proposal_id, voter_id, vote, **kwargs) -> bool:
        # Votar en propuesta (solo miembros del comité)
        
    def get_consensus_result(self, proposal_id) -> Optional[Dict]:
        # Obtener resultado final del consenso

# Orchestrator mejorado con VRF
class VRFOrchestrator(Orchestrator):
    def orchestrate_round(self, round_number, participating_clients):
        # 1. Selección VRF de comité
        # 2. Entrenamiento local estándar
        # 3. Agregación MI (Fase 2)
        # 4. Validación por comité
        # 5. Consenso y actualización
```

## 🧪 Validación y Pruebas

### FASE 1 - Validación de Compresión
- ✅ Test de PowerSGD independiente
- ✅ Verificación de cálculo de diferencias ΔW
- ✅ Simulación federada completa con 3 clientes
- ✅ Compresión 11.77x con error de reconstrucción < 0.1%

### FASE 2 - Validación de Robustez
- ✅ Test de MI aggregation independiente
- ✅ Simulación con clientes honestos y maliciosos (3+2)
- ✅ Detección automática de ataques de ruido y aleatorios
- ✅ Filtrado exitoso de contribuciones maliciosas

### FASE 3 - Validación de Consenso
- ✅ Test de selección VRF independiente
- ✅ Test de votación y consenso por comité
- ✅ Simulación VRF-FL completa con validación distribuida
- ✅ Integración exitosa de las 3 fases (Compresión + MI + VRF)

```bash
# Ejecutar todas las pruebas
python examples/simple_simulation.py        # FASE 1
python examples/phase2_mi_simulation.py     # FASE 2
python examples/phase3_vrf_simulation.py    # FASE 3

# Salida esperada FASE 3:
# 🎊 ALL PHASE 3 TESTS PASSED!
# ✅ VRF consensus successfully integrated with FL pipeline
# ✅ Committee-based validation working correctly
# ✅ Decentralized decision making functional
```

## 📋 Próximas Fases

### FASE 4 - Almacenamiento IPFS
- `IPFSStorageProvider` para almacenamiento distribuido real
- Integración con CIDs para referencias de modelo
- Descentralización completa del almacenamiento

### FASE 5 - Blockchain Real
- `EthereumConnector` con `web3.py`
- Contratos inteligentes para FL
- Producción en mainnet/testnet

### FASE 6 - Optimización IoT
- Algoritmos específicos para dispositivos con recursos limitados
- Compresión adaptativa según capacidad del dispositivo
- Protocolos de comunicación eficientes

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🔗 Enlaces Útiles

- [Documentación de Arquitectura](docs/LLM_CONTEXT.md)
- [Ejemplos](examples/)
- [Tests](tests/)

---

**Flare** - Entrenamiento Federado robusto, eficiente y descentralizado para la era IoT 🌟