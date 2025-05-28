import pytest
from flare.core import FlareConfig

def test_flare_config_set_get():
    config = FlareConfig()
    config.set("key1", "value1")
    config.set("key2", 123)
    assert config.get("key1") == "value1"
    assert config.get("key2") == 123
    assert config.get("non_existent_key") is None
    assert config.get("non_existent_key", "default") == "default"

def test_flare_config_get_required():
    config = FlareConfig({"req_key": "req_val"})
    assert config.get_required("req_key") == "req_val"
    with pytest.raises(ValueError):
        config.get_required("missing_key")

def test_flare_config_all_and_copy():
    initial_dict = {"a": 1, "b": 2}
    config1 = FlareConfig(initial_dict)
    assert config1.all() == initial_dict

    config2 = config1.copy()
    assert config2.all() == initial_dict
    config2.set("c", 3)
    assert config1.all() == initial_dict # Ensure original is not modified
    assert "c" in config2.all()

def test_flare_config_str():
    config = FlareConfig({"name": "test_config", "version": 1.0})
    assert str(config) == str({"name": "test_config", "version": 1.0})