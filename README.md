<div align="center">
  <img src="https://i.imgur.com/6vQYSqr.png">
</div>

# Flare: Federated Learning with Blockchain and IoT Focus

**Flare** es una biblioteca modular de Python para simular e implementar sistemas de **Entrenamiento Federado (FL)** que aprovechan **blockchain** para coordinación, confianza y registro, con un fuerte énfasis en la eficiencia y aplicabilidad para dispositivos **IoT**.

## 📈 Estado del Proyecto

| 🎯 Progreso | 🧪 Tests | 📖 Docs | 🚀 Ejemplos | 🏗️ Componentes |
|-------------|----------|----------|-------------|-----------------|
| **62%** (31/50) | ✅ Passing | ✅ Completa | ✅ 6 Ejemplos | ✅ 5 Fases |

**Últimos Logros**:
- ✅ **Sistema de Simulación Avanzado** (FASE 5) - Experimentación científica completa
- ✅ **Builder Pattern** (FASE 4) - Reducción 60% código de configuración  
- ✅ **VRF Consensus** (FASE 3) - Consenso descentralizado verificable
- ✅ **MI Robusta** (FASE 2) - Filtrado 40% clientes maliciosos
- ✅ **PowerSGD** (FASE 1) - Compresión 11.77x con <0.1% error

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

# Sistema de Simulación Avanzado (FASE 5) - Nuevo! 🆕
python examples/simulation_example.py
```

## 🏗️ Arquitectura Modular

### Módulos Principales

- **`flare.core`**: Clases base (`FlareConfig`, `FlareNode`, `RoundContext`)
- **`flare.models`**: Adaptadores de modelos (`ModelAdapter`, `PyTorchModelAdapter`)
- **`flare.compression`**: Compresores (`PowerSGDCompressor`, `ZlibCompressor`)
- **`flare.consensus`**: Mecanismos de consenso (`VRFConsensus`)
- **`flare.blockchain`**: Conectores blockchain (`MockChainConnector`)
- **`flare.storage`**: Proveedores de almacenamiento (`InMemoryStorageProvider`)
- **`flare.federation`**: Lógica FL principal (`Client`, `Orchestrator`, `AggregationStrategy`)
- **`flare.builder`**: Patrón Builder (`OrchestratorBuilder`, `ClientBuilder`)
- **`flare.simulation`**: Sistema de simulación avanzado (`Scenario`, `MetricsCollector`)

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

## 🗺️ Hoja de Ruta - Progreso del Proyecto

### ✅ FASE 0 - Infraestructura Base (Completada)
- [x] **Arquitectura Modular**: Interfaces abstractas para todos los componentes
- [x] **Mock Implementations**: Componentes de prueba para simulación rápida
- [x] **Core Classes**: `FlareConfig`, `FlareNode`, `RoundContext`
- [x] **Basic FL Pipeline**: Cliente y Orquestador básicos funcionales
- [x] **Testing Setup**: Estructura de pruebas con pytest

### ✅ FASE 1 - Compresión Eficiente (Completada)
- [x] **PowerSGDCompressor**: Compresión de bajo rango con iteración de potencia
- [x] **FederatedClient**: Cliente que computa ΔW (diferencias de pesos)
- [x] **PyTorchModelAdapter**: Adaptador completo para modelos PyTorch
- [x] **Device Management**: Soporte para CPU/GPU automático
- [x] **Compression Metrics**: Evaluación de ratios y errores de compresión
- [x] **Phase 1 Simulation**: Demostración completa con métricas

**Logros**: 11.77x compresión con <0.1% error de reconstrucción

### ✅ FASE 2 - Agregación Robusta (Completada)
- [x] **MIAggregationStrategy**: Filtrado de clientes maliciosos usando Mutual Information
- [x] **Malicious Detection**: Detección automática de contribuciones anómalas
- [x] **Attack Simulation**: Tipos de ataques (ruido, aleatorio, adversarial)
- [x] **Byzantine Tolerance**: FL resistente a clientes comprometidos
- [x] **MI Metrics**: Evaluación de eficacia del filtrado
- [x] **Phase 2 Simulation**: Demostración con clientes maliciosos

**Logros**: Filtrado exitoso de 40% de clientes maliciosos manteniendo convergencia

### ✅ FASE 3 - Consenso VRF (Completada)
- [x] **VRFConsensus**: Mecanismo de consenso basado en Verifiable Random Function
- [x] **Committee Selection**: Selección determinística y verificable de validadores
- [x] **Distributed Validation**: Validación descentralizada de modelos agregados
- [x] **Threshold Consensus**: Consenso por umbral configurable
- [x] **Early Detection**: Detección temprana de consenso para eficiencia
- [x] **Phase 3 Simulation**: Pipeline completo FL+MI+VRF

**Logros**: Consenso descentralizado con selección verificable de comités

### ✅ FASE 4 - Builder Pattern (Completada)
- [x] **OrchestratorBuilder**: Constructor fluido para orquestadores
- [x] **ClientBuilder**: Constructor fluido para clientes
- [x] **Intelligent Defaults**: Valores por defecto inteligentes
- [x] **Validation System**: Validación automática de componentes requeridos
- [x] **API Simplificada**: Reducción significativa de código boilerplate
- [x] **Documentation**: Guías completas del patrón Builder

**Logros**: Reducción del 60% en líneas de código para configuración

### ✅ FASE 5 - Sistema de Simulación Avanzado (Completada) 🆕
- [x] **Scenario System**: Definición declarativa de escenarios de simulación
- [x] **Metrics Collection**: Recolección automatizada de métricas con metadatos
- [x] **Run Management**: Gestión completa del ciclo de vida de experimentos
- [x] **Visualization Tools**: Generación automática de gráficos y reportes
- [x] **Export/Import**: Exportación de resultados para análisis externo
- [x] **Batch Execution**: Ejecución de múltiples experimentos en lote

**Componentes Implementados**:
- `flare.simulation.scenario`: Sistema de escenarios con ScenarioBuilder
- `flare.simulation.metrics`: Recolección de métricas tipadas
- `flare.simulation.run_manager`: Gestión de experimentos y persistencia
- `flare.simulation.visualization`: Generación automática de visualizaciones

**Logros**: Sistema completo de experimentación científica con reproducibilidad

---

### 🔄 FASE 6 - IPFS Storage (En Desarrollo)
- [ ] **IPFSStorageProvider**: Almacenamiento distribuido usando IPFS
- [ ] **Content Addressing**: Referencias usando CIDs para inmutabilidad
- [ ] **Decentralized Storage**: Eliminación de puntos únicos de falla
- [ ] **Model Versioning**: Versionado automático de modelos usando DAG
- [ ] **Bandwidth Optimization**: Transferencia eficiente para IoT

### 🔄 FASE 7 - Blockchain Real (Planificada)
- [ ] **EthereumConnector**: Integración con Web3.py para Ethereum
- [ ] **Smart Contracts**: Contratos para coordinación y gobernanza FL
- [ ] **Transaction Management**: Gestión eficiente de transacciones
- [ ] **Gas Optimization**: Optimización de costos para IoT
- [ ] **Multi-chain Support**: Soporte para múltiples blockchains

### 🔄 FASE 8 - Optimización IoT (Planificada)
- [ ] **Adaptive Compression**: Compresión adaptativa según recursos del dispositivo
- [ ] **Edge Computing**: Optimizaciones para computación en el borde
- [ ] **Battery Awareness**: Gestión eficiente de energía
- [ ] **Network Protocols**: Protocolos optimizados para IoT (CoAP, MQTT)
- [ ] **Resource Monitoring**: Monitoreo de recursos en tiempo real

### 🔄 FASE 9 - Seguridad Avanzada (Planificada)
- [ ] **Differential Privacy**: Preservación de privacidad en agregación
- [ ] **Secure Aggregation**: Agregación criptográfica de actualizaciones
- [ ] **Zero-Knowledge Proofs**: Verificación sin exposición de datos
- [ ] **Homomorphic Encryption**: Computación sobre datos cifrados
- [ ] **Attestation**: Verificación de integridad en dispositivos IoT

### 🔄 FASE 10 - Producción y Escalabilidad (Planificada)
- [ ] **Load Balancing**: Distribución de carga en orquestadores
- [ ] **Auto-scaling**: Escalado automático según demanda
- [ ] **Monitoring Dashboard**: Panel de control en tiempo real
- [ ] **CI/CD Pipeline**: Integración y despliegue continuo
- [ ] **Performance Benchmarks**: Benchmarks estandarizados para comparación

---

## 📊 Métricas de Progreso

| Fase | Estado | Componentes | Tests | Documentación | Ejemplos |
|------|--------|-------------|-------|---------------|----------|
| 0 - Base | ✅ | 15/15 | ✅ | ✅ | ✅ |
| 1 - Compresión | ✅ | 3/3 | ✅ | ✅ | ✅ |
| 2 - MI Robusta | ✅ | 2/2 | ✅ | ✅ | ✅ |
| 3 - VRF Consenso | ✅ | 2/2 | ✅ | ✅ | ✅ |
| 4 - Builder Pattern | ✅ | 4/4 | ✅ | ✅ | ✅ |
| 5 - Simulación | ✅ | 5/5 | ✅ | ✅ | ✅ |
| 6 - IPFS | 🔄 | 0/4 | ❌ | ❌ | ❌ |
| 7 - Blockchain | 📋 | 0/5 | ❌ | ❌ | ❌ |
| 8 - IoT | 📋 | 0/5 | ❌ | ❌ | ❌ |
| 9 - Seguridad | 📋 | 0/5 | ❌ | ❌ | ❌ |
| 10 - Producción | 📋 | 0/5 | ❌ | ❌ | ❌ |

**Progreso Total: 31/50 componentes principales (62%)**

### 🎯 Próximos Hitos

1. **Q1 2024**: Completar FASE 6 (IPFS Storage)
2. **Q2 2024**: Implementar FASE 7 (Blockchain Real)  
3. **Q3 2024**: Optimizaciones IoT (FASE 8)
4. **Q4 2024**: Seguridad Avanzada (FASE 9)
5. **2025**: Preparación para Producción (FASE 10)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Revisa la hoja de ruta para identificar áreas de trabajo
2. Abre un issue para discutir cambios importantes
3. Sigue las convenciones de código establecidas
4. Incluye tests para nuevas funcionalidades
5. Actualiza la documentación correspondiente

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Reconocimientos

- Inspirado en los papers de BEFL (Blockchain-Enabled Federated Learning)
- Algoritmos de compresión basados en PowerSGD
- Implementación de VRF basada en especificaciones criptográficas estándar