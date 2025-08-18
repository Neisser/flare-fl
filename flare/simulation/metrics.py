"""
Metrics Module for Flare Simulations

Gestiona la recolección, almacenamiento y análisis de métricas durante las simulaciones FL.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class MetricType(Enum):
    """Tipos de métricas soportadas."""

    LOSS = auto()
    ACCURACY = auto()
    COMMUNICATION_BYTES = auto()
    COMPUTATION_TIME = auto()
    ROUND_TIME = auto()
    MODEL_SIZE = auto()
    COMPRESSION_RATIO = auto()
    CONSENSUS_TIME = auto()
    BLOCKCHAIN_TIME = auto()
    STORAGE_TIME = auto()
    CUSTOM = auto()


@dataclass
class MetricValue:
    """Representa un valor de métrica con su timestamp."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Series temporal de valores para una métrica específica."""

    metric_type: MetricType
    name: str
    values: List[MetricValue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_value(
        self, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Añade un nuevo valor a la serie."""
        self.values.append(MetricValue(value=value, metadata=metadata or {}))

    def get_values(self) -> List[float]:
        """Retorna la lista de valores sin timestamps."""
        return [v.value for v in self.values]

    def get_timestamps(self) -> List[datetime]:
        """Retorna la lista de timestamps."""
        return [v.timestamp for v in self.values]

    def get_latest(self) -> Optional[float]:
        """Retorna el último valor registrado."""
        return self.values[-1].value if self.values else None

    def get_statistics(self) -> Dict[str, float]:
        """Calcula estadísticas básicas de la serie."""
        if not self.values:
            return {}

        values = self.get_values()
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "latest": float(values[-1]),
        }


class MetricsCollector:
    """Recolecta y gestiona métricas durante una simulación."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.metrics: Dict[str, MetricSeries] = {}
        self.start_time = datetime.now()

    def register_metric(
        self,
        metric_type: MetricType,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra una nueva métrica para seguimiento."""
        if name in self.metrics:
            raise ValueError(f"Metric {name} already registered")

        self.metrics[name] = MetricSeries(
            metric_type=metric_type, name=name, metadata=metadata or {}
        )

    def record_value(
        self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registra un nuevo valor para una métrica."""
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not registered")

        self.metrics[name].add_value(value, metadata)

    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Retorna una métrica por nombre."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Retorna todas las métricas registradas."""
        return self.metrics

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calcula estadísticas para todas las métricas."""
        return {name: series.get_statistics() for name, series in self.metrics.items()}

    def save_metrics(self, path: Union[str, Path]) -> None:
        """Guarda todas las métricas en formato JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "scenario_name": self.scenario_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "metrics": {
                name: {
                    "type": series.metric_type.name,
                    "values": [
                        {
                            "value": v.value,
                            "timestamp": v.timestamp.isoformat(),
                            "metadata": v.metadata,
                        }
                        for v in series.values
                    ],
                    "metadata": series.metadata,
                    "statistics": series.get_statistics(),
                }
                for name, series in self.metrics.items()
            },
        }

        with open(path / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

    def load_metrics(self, path: Union[str, Path]) -> None:
        """Carga métricas desde un archivo JSON."""
        path = Path(path)
        with open(path / "metrics.json", "r") as f:
            data = json.load(f)

        self.scenario_name = data["scenario_name"]
        self.start_time = datetime.fromisoformat(data["start_time"])

        for name, metric_data in data["metrics"].items():
            series = MetricSeries(
                metric_type=MetricType[metric_data["type"]],
                name=name,
                metadata=metric_data["metadata"],
            )

            for value_data in metric_data["values"]:
                series.add_value(
                    value=value_data["value"], metadata=value_data["metadata"]
                )

            self.metrics[name] = series
            self.metrics[name] = series
