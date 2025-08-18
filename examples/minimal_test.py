"""
Test mínimo para identificar el problema de corrupción JSON.
"""

from flare.simulation import MetricsCollector, MetricType, RunManager, Scenario


class MockOrchestrator:
    def get_communication_bytes(self):
        return 1000


class MockClient:
    def __init__(self, client_id):
        self.client_id = client_id


def main():
    print("🔍 Test mínimo del sistema de simulación...")

    # 1. Crear componentes
    orchestrator = MockOrchestrator()
    clients = [MockClient("client_1")]

    # 2. Crear escenario
    scenario = Scenario(
        name="minimal_test",
        description="Test mínimo",
        orchestrator=orchestrator,
        clients=clients,
        num_rounds=1,
        batch_size=32,
        learning_rate=0.01,
    )

    # 3. RunManager
    run_manager = RunManager("minimal_runs")

    # 4. Crear run
    print("📝 Creando run...")
    run_metadata = run_manager.create_run(scenario, "Test mínimo")
    print(f"✅ Run creado: {run_metadata.run_id}")

    # Verificar que existe
    metadata_path = run_manager.base_path / "metadata" / f"{run_metadata.run_id}.json"
    print(f"📁 Archivo: {metadata_path}")
    with open(metadata_path, "r") as f:
        content = f.read()
        print(f"📄 Contenido inicial ({len(content)} bytes): OK")

    # 5. Update status (aquí suele fallar)
    print("🔄 Actualizando status...")
    try:
        run_manager.update_run_status(run_metadata.run_id, "running")
        print("✅ Status actualizado a 'running'")

        # Verificar archivo después del update
        with open(metadata_path, "r") as f:
            content = f.read()
            print(f"📄 Contenido después update ({len(content)} bytes)")
            if len(content) == 0:
                print("❌ ¡Archivo se vació después del update!")
            else:
                print("✅ Archivo OK después del update")

    except Exception as e:
        print(f"❌ Error en update_run_status: {e}")

        # Verificar el estado del archivo
        with open(metadata_path, "r") as f:
            content = f.read()
            print(f"📄 Contenido después del error ({len(content)} bytes)")
        return

    # 6. Segundo update (completado)
    print("🔄 Segundo update...")
    try:
        run_manager.update_run_status(run_metadata.run_id, "completed")
        print("✅ Status actualizado a 'completed'")
    except Exception as e:
        print(f"❌ Error en segundo update: {e}")

    print("🎉 Test mínimo completado")


if __name__ == "__main__":
    main()
