"""
Visualization Module for Flare Simulations

Proporciona herramientas para visualizar y analizar resultados de simulaciones FL.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import MetricsCollector, MetricType
from .run_manager import RunManager, RunMetadata


class SimulationVisualizer:
    """Herramientas para visualizar resultados de simulaciones."""

    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager
        plt.style.use("seaborn")

    def plot_metric_evolution(
        self,
        run_id: str,
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plotea la evolución de una métrica a lo largo de las rondas."""
        run_path = self.run_manager.base_path / run_id
        with open(run_path / "metrics.json", "r") as f:
            metrics_data = json.load(f)

        if metric_name not in metrics_data["metrics"]:
            raise ValueError(f"Metric {metric_name} not found in run {run_id}")

        metric_data = metrics_data["metrics"][metric_name]
        values = [v["value"] for v in metric_data["values"]]
        timestamps = [
            datetime.fromisoformat(v["timestamp"]) for v in metric_data["values"]
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, values, marker="o", linestyle="-", linewidth=2)

        plt.title(title or f"{metric_name} Evolution")
        plt.xlabel("Time")
        plt.ylabel(metric_name)
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_metrics_comparison(
        self,
        run_ids: List[str],
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Compara una métrica entre múltiples corridas."""
        plt.figure(figsize=(12, 7))

        for run_id in run_ids:
            run_path = self.run_manager.base_path / run_id
            with open(run_path / "metrics.json", "r") as f:
                metrics_data = json.load(f)

            if metric_name not in metrics_data["metrics"]:
                print(f"Warning: Metric {metric_name} not found in run {run_id}")
                continue

            metric_data = metrics_data["metrics"][metric_name]
            values = [v["value"] for v in metric_data["values"]]
            timestamps = [
                datetime.fromisoformat(v["timestamp"]) for v in metric_data["values"]
            ]

            metadata = self.run_manager.get_run(run_id)
            label = f"{metadata.scenario_name} ({metadata.run_id[:8]})"

            plt.plot(
                timestamps, values, marker="o", linestyle="-", linewidth=2, label=label
            )

        plt.title(title or f"{metric_name} Comparison")
        plt.xlabel("Time")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_metric_distribution(
        self,
        run_ids: List[str],
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plotea la distribución de una métrica entre múltiples corridas."""
        plt.figure(figsize=(10, 6))

        data = []
        labels = []

        for run_id in run_ids:
            run_path = self.run_manager.base_path / run_id
            with open(run_path / "metrics.json", "r") as f:
                metrics_data = json.load(f)

            if metric_name not in metrics_data["metrics"]:
                print(f"Warning: Metric {metric_name} not found in run {run_id}")
                continue

            metric_data = metrics_data["metrics"][metric_name]
            values = [v["value"] for v in metric_data["values"]]

            metadata = self.run_manager.get_run(run_id)
            label = f"{metadata.scenario_name} ({metadata.run_id[:8]})"

            data.append(values)
            labels.append(label)

        sns.boxplot(data=data, labels=labels)

        plt.title(title or f"{metric_name} Distribution")
        plt.xticks(rotation=45)
        plt.ylabel(metric_name)
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_metric_correlation(
        self,
        run_id: str,
        metric_names: List[str],
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plotea la correlación entre múltiples métricas de una corrida."""
        run_path = self.run_manager.base_path / run_id
        with open(run_path / "metrics.json", "r") as f:
            metrics_data = json.load(f)

        # Preparar datos
        data = {}
        for name in metric_names:
            if name not in metrics_data["metrics"]:
                print(f"Warning: Metric {name} not found in run {run_id}")
                continue

            metric_data = metrics_data["metrics"][name]
            values = [v["value"] for v in metric_data["values"]]
            data[name] = values

        # Crear matriz de correlación
        df = pd.DataFrame(data)
        corr_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)

        plt.title(title or "Metric Correlations")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def generate_run_report(self, run_id: str, output_path: Union[str, Path]) -> None:
        """Genera un reporte completo de una corrida con gráficos."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Cargar datos
        run_path = self.run_manager.base_path / run_id
        with open(run_path / "metrics.json", "r") as f:
            metrics_data = json.load(f)

        metadata = self.run_manager.get_run(run_id)

        # Generar gráficos para cada métrica
        for metric_name in metrics_data["metrics"].keys():
            self.plot_metric_evolution(
                run_id=run_id,
                metric_name=metric_name,
                save_path=output_path / f"{metric_name}_evolution.png",
            )

        # Generar resumen en HTML
        html_content = f"""
        <html>
        <head>
            <title>Run Report - {metadata.run_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Run Report</h1>
                <p><strong>Run ID:</strong> {metadata.run_id}</p>
                <p><strong>Scenario:</strong> {metadata.scenario_name}</p>
                <p><strong>Timestamp:</strong> {metadata.timestamp}</p>
                <p><strong>Status:</strong> {metadata.status}</p>
                <p><strong>Description:</strong> {metadata.description}</p>
                <p><strong>Tags:</strong> {", ".join(metadata.tags)}</p>
            </div>
            
            <h2>Metrics Evolution</h2>
            <div class="metrics">
        """

        for metric_name in metrics_data["metrics"].keys():
            html_content += f"""
                <div class="metric-card">
                    <h3>{metric_name}</h3>
                    <img src="{metric_name}_evolution.png" alt="{metric_name} evolution">
                    <p><strong>Statistics:</strong></p>
                    <ul>
            """

            stats = metrics_data["metrics"][metric_name]["statistics"]
            for stat_name, value in stats.items():
                html_content += f"<li>{stat_name}: {value:.4f}</li>"

            html_content += """
                    </ul>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        with open(output_path / "report.html", "w") as f:
            f.write(html_content)
