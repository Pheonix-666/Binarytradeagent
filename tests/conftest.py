"""Shared pytest fixtures."""

from pathlib import Path
import pytest
import yaml

@pytest.fixture
def valid_config_dict():
    """Return a dictionary representing a valid default configuration."""
    with open("config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

@pytest.fixture
def temp_config_file(tmp_path, valid_config_dict):
    """Factory fixture to create temporary YAML configuration files."""
    def _create_config(**overrides):
        cfg = dict(valid_config_dict)
        for key, val in overrides.items():
            if "." in key:
                parts = key.split(".")
                curr = cfg
                for p in parts[:-1]:
                    curr = curr[p]
                curr[parts[-1]] = val
            else:
                cfg[key] = val
        file_path = tmp_path / "test_config.yaml"
        with open(file_path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh)
        return file_path

    return _create_config

