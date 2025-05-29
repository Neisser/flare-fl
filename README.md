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

### 🎯 Algoritmo de Mutual Information (FASE 2)

La estrategia `MIAggregationStrategy` implementa:

1. **Cálculo de Firmas de Modelo**: Extrae estadísticas de pesos y pseudo-predicciones
2. **MI Pairwise**: Computa Mutual Information entre firmas de modelos usando `sklearn`
3. **Detección de Outliers**: Identifica modelos con patrones de MI anómalos
4. **Agregación Filtrada**: Solo agrega actualizaciones de clientes confiables

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

### Ejecutar Simulaciones

```bash
# FASE 1: Compresión PowerSGD
python examples/simple_simulation.py

# FASE 2: Agregación con MI + Clientes maliciosos
python examples/phase2_mi_simulation.py
```

## 🏗️ Arquitectura Modular

### Módulos Principales

- **`flare.core`**: Clases base (`FlareConfig`, `FlareNode`, `RoundContext`)
- **`flare.models`**: Adaptadores de modelos (`ModelAdapter`, `PyTorchModelAdapter`)
- **`flare.compression`**: Compresores (`PowerSGDCompressor`, `GzipCompressor`)
- **`flare.federation`**: Lógica FL (`FederatedClient`, `Orchestrator`, `MIAggregationStrategy`)
- **`flare.blockchain`**: Conectores blockchain (`MockChainConnector`)
- **`flare.storage`**: Proveedores de almacenamiento (`InMemoryStorageProvider`)

### Interfaces Clave - FASE 2

```python
# Estrategia de agregación con MI
class MIAggregationStrategy(AggregationStrategy):
    def aggregate(self, local_model_updates, client_data_sizes=None, 
                  previous_global_weights=None, **kwargs) -> ModelWeights:
        # Filtrar modelos maliciosos usando MI
        trusted_indices = self._filter_malicious_updates(
            model_updates, test_data, global_weights
        )
        # Agregar solo actualizaciones confiables
        return self._weighted_average(trusted_updates, data_sizes)

# Cliente malicioso para pruebas
class MaliciousClient(FederatedClient):
    def train_local(self, round_context, epochs, learning_rate):
        # Simular diferentes tipos de ataques
        if self.malicious_type == "noise":
            # Agregar ruido a actualizaciones legítimas
        elif self.malicious_type == "random":
            # Enviar pesos completamente aleatorios
        elif self.malicious_type == "opposite":
            # Invertir dirección de actualizaciones
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

```bash
# Ejecutar todas las pruebas
python examples/simple_simulation.py        # FASE 1
python examples/phase2_mi_simulation.py     # FASE 2

# Salida esperada FASE 2:
# 🎉 ALL PHASE 2 TESTS PASSED!
# ✅ MI-based aggregation successfully filtered malicious clients
```

## 📋 Próximas Fases

### FASE 3 - Consenso VRF
- Implementar `VRFConsensus` para selección de comité
- Votación de modelos en blockchain
- Consenso verificable y descentralizado

### FASE 4 - Almacenamiento IPFS
- `IPFSStorageProvider` para almacenamiento distribuido real
- Integración con CIDs para referencias de modelo

### FASE 5 - Blockchain Real
- `EthereumConnector` con `web3.py`
- Contratos inteligentes para FL
- Producción en mainnet/testnet

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

**Flare** - Entrenamiento Federado robusto y eficiente para la era IoT 🌟