"""
Debug específico para el problema de Scenario con mocks.
"""

from flare.simulation import RunManager, Scenario


class MockOrchestrator:
    def __init__(self):
        self.rounds_completed = 0


class MockClient:
    def __init__(self, client_id):
        self.client_id = client_id


def test_scenario_creation():
    print("🔍 Testing Scenario creation...")

    # 1. Crear componentes mock
    orchestrator = MockOrchestrator()
    clients = [MockClient(f"client_{i}") for i in range(2)]

    # 2. Crear escenario
    try:
        scenario = Scenario(
            name="test_scenario",
            description="Test con mocks",
            orchestrator=orchestrator,
            clients=clients,
            num_rounds=3,
            batch_size=32,
            learning_rate=0.01,
            tags=["test"],
        )
        print("✅ Scenario creado")

        # 3. Probar to_dict
        try:
            scenario_dict = scenario.to_dict()
            print("✅ to_dict() funciona")
            print(f"📄 Keys: {list(scenario_dict.keys())}")
        except Exception as e:
            print(f"❌ Error en to_dict(): {e}")
            import traceback

            traceback.print_exc()
            return

        # 4. Probar RunManager.create_run pero SIN save_metadata
        rm = RunManager("debug_scenario_runs")

        # Modificar temporalmente el método create_run
        original_create_run = rm.create_run

        def create_run_no_scenario_save(scenario, description="", tags=None):
            from flare.simulation.run_manager import RunMetadata

            metadata = RunMetadata(
                scenario_name=scenario.name,
                description=description or scenario.description,
                tags=tags or scenario.tags,
                parameters=scenario.parameters,
            )

            # Crear directorio para la corrida
            run_path = rm.base_path / metadata.run_id
            run_path.mkdir(parents=True)

            # Guardar metadata inicial
            rm._save_metadata(metadata)

            # NO guardar configuración del escenario
            # scenario.save_metadata(run_path)  # <-- Comentado

            return metadata

        rm.create_run = create_run_no_scenario_save

        try:
            run_metadata = rm.create_run(scenario, "Test sin scenario save")
            print(
                f"✅ create_run sin scenario.save_metadata funciona: {run_metadata.run_id}"
            )
        except Exception as e:
            print(f"❌ Error en create_run sin scenario save: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ Error creando scenario: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_scenario_creation()
