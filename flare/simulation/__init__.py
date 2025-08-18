"""
Flare Simulation Module

Este módulo proporciona herramientas para gestionar, ejecutar y analizar simulaciones
de Federated Learning de manera sistemática y reproducible.

Estructura:
- Scenario: Define un escenario de simulación (configuración, participantes, etc.)
- Metrics: Gestiona métricas y logging durante la simulación
- RunManager: Gestiona el almacenamiento y versionado de corridas
- Visualization: Herramientas para visualizar resultados
"""

from .metrics import MetricsCollector, MetricType
from .run_manager import RunManager, RunMetadata
from .scenario import Scenario, ScenarioBuilder
from .visualization import SimulationVisualizer

__all__ = [
    "Scenario",
    "ScenarioBuilder",
    "MetricsCollector",
    "MetricType",
    "RunManager",
    "RunMetadata",
    "SimulationVisualizer",
]
