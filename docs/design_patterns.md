## 🧩 1. **Modularidad (Configurabilidad y Extensibilidad)**

| Patrón                         | Aplicación                                                                              | Ejemplo en FL + Blockchain                                        |
| ------------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Strategy**                   | Para seleccionar dinámicamente el algoritmo de agregación, consenso, compresión, etc.   | `AggregationStrategy`, `ConsensusStrategy`, `CompressionStrategy` |
| **Factory Method**             | Para crear objetos de agregadores, nodos, contratos según configuración.                | `AggregatorFactory`, `BlockchainClientFactory`                    |
| **Abstract Factory**           | Para grupos de familias de componentes (por ejemplo: toda la capa blockchain).          | `BlockchainEnvironmentFactory`: EthereumFactory, FabricFactory    |
| **Builder**                    | Para construir objetos complejos paso a paso, como un `FederatedRound` o `TrainingJob`. | `TrainingJobBuilder`                                              |
| **Decorator**                  | Para agregar privacidad (DP, HE) o logging a una agregación sin modificarla.            | `DPDecorator(Aggregator)`, `LoggingDecorator(Node)`               |
| **Plugin (o Service Locator)** | Para permitir añadir módulos como "plugins" desde archivos externos.                    | Agregadores definidos como `.py` o `.so` externos                 |

---

## 🎮 2. **Simulación y Orquestación**

| Patrón        | Aplicación                                                                                      | Ejemplo en simulación                                   |
| ------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Observer**  | Para notificar a los nodos, blockchain o UI cuando hay un nuevo modelo, evento, etc.            | `ModelUpdateObserver`, `TrainingStatusObserver`         |
| **Command**   | Para encapsular tareas (e.g. entrenar, subir hash, recibir recompensa).                         | `TrainCommand`, `UploadModelCommand`, `EvaluateCommand` |
| **Mediator**  | Para orquestar la comunicación entre nodos, agregador y blockchain sin acoplarlos directamente. | `FederatedCoordinator`, `BlockchainMediator`            |
| **State**     | Para manejar el estado de los nodos (idle, training, uploading, waiting).                       | `NodeState: Idle → Training → Uploading`                |
| **Prototype** | Para clonar nodos o configuraciones rápidamente al simular cientos.                             | `Node.clone()`, `SimulationScenario.clone()`            |

---

## 🔐 3. **Seguridad, Privacidad y Contratos**

| Patrón                      | Aplicación                                                                         | Ejemplo en seguridad                                  |
| --------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Proxy**                   | Para interponer lógica adicional como verificación de firmas, validación, cifrado. | `SecureAggregatorProxy`, `BlockchainProxy`            |
| **Chain of Responsibility** | Para manejar validaciones múltiples de nodos, modelos o transacciones.             | `ModelValidator → SignatureChecker → OutlierDetector` |

---

## 📈 4. **Monitorización, Métricas y Logs**

| Patrón              | Aplicación                                                                             | Ejemplo                                                                     |
| ------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Singleton**       | Para tener un logger, monitor o repositorio de métricas global.                        | `TrainingLogger.getInstance()`, `GlobalMonitor`                             |
| **Template Method** | Para definir esqueleto de ejecución de una ronda FL con pasos fijos y otros variables. | `FederatedRound.run()` con pasos definidos, y `hook()` para personalización |

---

## 📦 5. **Gestión de Componentes y Versionamiento**

| Patrón      | Aplicación                                                                                      | Ejemplo                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Adapter** | Para permitir que nuevos modelos, clientes o blockchains encajen con las interfaces existentes. | `PolkadotAdapter`, `LegacyClientAdapter`                                     |
| **Bridge**  | Para desacoplar la abstracción del modelo federado del backend FL (Torch/TensorFlow).           | `ModelBridge` permite usar PyTorch o TensorFlow sin cambiar código principal |

---

## ✅ Recomendaciones prácticas

* Usa **Strategy + Factory + Decorator** como núcleo de la arquitectura para mantener una alta extensibilidad.
* Mantén **cada módulo desacoplado y versionado** (por ejemplo: `aggregators/fedavg_v1.py`, `fedprox_v2.py`).
* Aplica el patrón **Observer + Mediator** para que la orquestación y simulación sean dinámicas.
* Asegura que **todos los objetos configurables** (consenso, agregación, privacidad) se definan en archivos `.yaml` o `.json` para permitir simulaciones automatizadas.
