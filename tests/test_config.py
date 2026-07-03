"""
tests/test_config.py

Unit tests for src/utils/config.py (M0 — Project Bootstrap).

Covers IMP-01 M0 Definition of Done:
- "Config loader raises ValueError with an informative message for any
  missing required field."
Also covers V-INV-001 precondition: config schema consistency across
conditions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ConfigValidationError,
    REQUIRED_BASE_FIELDS,
    load_condition_config,
    load_config,
    validate_schema,
)
from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT


# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def valid_config_dict() -> dict:
    """A minimal but fully schema-valid config dict for isolated tests."""
    return {
        "encoder": {
            "architecture": "ts2vec",
            "input_dim": 7,
            "hidden_dim": 64,
            "output_dim": 64,
            "depth": 10,
            "kernel_size": 3,
            "mask_ratio": 0.5,
        },
        "fusion": {"output_dim": 256, "projection_seed": 42},
        "window": {"size": 48, "stride": 1, "n_features": 7},
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "max_epochs": 50,
            "early_stopping_patience": 10,
        },
        "clustering": {
            "algorithm": "hdbscan",
            "distance_metric": "euclidean",
            "min_cluster_size_grid": [50, 100, 200],
            "min_samples_grid": [5, 10, 20],
            "max_clusters": 8,
        },
        "dataset": {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        },
        "seeds": [42, 123, 456, 789, 1024],
    }


@pytest.fixture
def tmp_config_file(tmp_path: Path, valid_config_dict: dict) -> Path:
    """Write a valid config dict to a temp YAML file and return its path."""
    path = tmp_path / "test_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)
    return path


# --- validate_schema tests ---------------------------------------------------

class TestValidateSchema:
    def test_valid_config_passes(self, valid_config_dict: dict) -> None:
        """A fully populated config must not raise."""
        validate_schema(valid_config_dict)

    def test_missing_single_field_raises(self, valid_config_dict: dict) -> None:
        """Removing one required field must raise ConfigValidationError."""
        del valid_config_dict["fusion"]["projection_seed"]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_schema(valid_config_dict)
        assert "fusion.projection_seed" in str(exc_info.value)

    def test_missing_multiple_fields_lists_all(self, valid_config_dict: dict) -> None:
        """Error message must list every missing field, not just the first."""
        del valid_config_dict["encoder"]["depth"]
        del valid_config_dict["training"]["batch_size"]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_schema(valid_config_dict)
        message = str(exc_info.value)
        assert "encoder.depth" in message
        assert "training.batch_size" in message

    def test_missing_top_level_section_raises(self, valid_config_dict: dict) -> None:
        """Removing an entire top-level section must raise informatively."""
        del valid_config_dict["clustering"]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_schema(valid_config_dict)
        assert "clustering" in str(exc_info.value)

    def test_error_message_is_informative(self, valid_config_dict: dict) -> None:
        """DoD requires an 'informative' message, not a bare KeyError."""
        del valid_config_dict["seeds"]
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_schema(valid_config_dict)
        message = str(exc_info.value)
        # Must reference the missing field and point somewhere actionable.
        assert "seeds" in message
        assert "DS-03" in message or "required" in message.lower()


# --- load_config tests --------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_config_file: Path) -> None:
        config = load_config(tmp_config_file)
        assert config["encoder"]["architecture"] == "ts2vec"
        assert config["seeds"] == [42, 123, 456, 789, 1024]

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(missing_path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(empty_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        list_path = tmp_path / "list.yaml"
        list_path.write_text("- 1\n- 2\n- 3\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(list_path)

    def test_invalid_config_raises_validation_error(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        del valid_config_dict["window"]
        path = tmp_path / "invalid.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(valid_config_dict, f)
        with pytest.raises(ConfigValidationError):
            load_config(path)

    def test_validate_false_skips_validation(
        self, tmp_path: Path, valid_config_dict: dict
    ) -> None:
        del valid_config_dict["window"]
        path = tmp_path / "partial.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(valid_config_dict, f)
        # Must NOT raise, since validation is explicitly disabled.
        config = load_config(path, validate=False)
        assert "window" not in config


# --- Real repository config files ---------------------------------------------

class TestRealConfigFiles:
    """
    Validates the actual configs/ files shipped in this repository,
    not synthetic fixtures. This is the test that would fail if
    base.yaml or any experiment_*.yaml drifts from the required schema.
    """

    def test_base_config_is_valid(self) -> None:
        config = load_config(CONFIGS_DIR / "base.yaml")
        assert config["fusion"]["output_dim"] == 256
        assert config["window"]["size"] == 48
        assert config["seeds"] == [42, 123, 456, 789, 1024]

    @pytest.mark.parametrize(
        "filename",
        [
            "experiment_1tf.yaml",
            "experiment_2tf.yaml",
            "experiment_3tf.yaml",
            "experiment_4tf.yaml",
            "experiment_bl_15m.yaml",
            "experiment_bl_4h.yaml",
            "experiment_bl_1d.yaml",
        ],
    )
    def test_all_seven_condition_configs_merge_and_validate(self, filename: str) -> None:
        """
        Each of the 7 TS2Vec condition configs (DS-03 §4 / IMP-01 v1.1)
        must merge cleanly over base.yaml and pass full schema validation.
        """
        merged = load_condition_config(
            CONFIGS_DIR / filename, CONFIGS_DIR / "base.yaml"
        )
        assert "condition" in merged
        assert merged["condition"]["name"]
        assert isinstance(merged["condition"]["active_timeframes"], list)
        assert len(merged["condition"]["active_timeframes"]) >= 1

    def test_condition_configs_only_differ_in_active_timeframes(self) -> None:
        """
        INV-001: only the set of active temporal resolutions may differ
        between conditions. Verify every merged condition config is
        identical to base.yaml except for the `condition` block.
        """
        base = load_config(CONFIGS_DIR / "base.yaml")
        filenames = [
            "experiment_1tf.yaml",
            "experiment_2tf.yaml",
            "experiment_3tf.yaml",
            "experiment_4tf.yaml",
            "experiment_bl_15m.yaml",
            "experiment_bl_4h.yaml",
            "experiment_bl_1d.yaml",
        ]
        for filename in filenames:
            merged = load_condition_config(
                CONFIGS_DIR / filename, CONFIGS_DIR / "base.yaml"
            )
            for key in base:
                if key == "condition":
                    continue
                assert merged[key] == base[key], (
                    f"{filename} unexpectedly overrides '{key}' — "
                    f"INV-001 requires only condition.active_timeframes to differ."
                )

    def test_seven_unique_ts2vec_conditions_have_distinct_names(self) -> None:
        """Sanity check on DS-03 §3.6: exactly 7 unique condition names."""
        filenames = [
            "experiment_1tf.yaml",
            "experiment_2tf.yaml",
            "experiment_3tf.yaml",
            "experiment_4tf.yaml",
            "experiment_bl_15m.yaml",
            "experiment_bl_4h.yaml",
            "experiment_bl_1d.yaml",
        ]
        names = set()
        for filename in filenames:
            config = load_config(
                CONFIGS_DIR / filename,
                required_fields=("condition.name", "condition.active_timeframes"),
            )
            names.add(config["condition"]["name"])
        assert len(names) == 7, f"Expected 7 unique condition names, got {names}"

    @pytest.mark.parametrize("filename", ["baseline_hmm.yaml", "baseline_kmpca.yaml"])
    def test_external_baseline_configs_load(self, filename: str) -> None:
        """
        The 2 external baseline configs (HMM, KM-PCA) intentionally do
        NOT carry the full base.yaml schema (IMP-01 v1.1 M10.5).
        """
        config = load_config(
            CONFIGS_DIR / filename,
            required_fields=("baseline.name", "baseline.method", "baseline.seeds"),
        )
        assert config["baseline"]["seeds"] == [42, 123, 456, 789, 1024]

    def test_all_config_files_exist(self) -> None:
        """
        IMP-01 v1.1 checklist: configs/experiment_{condition}.yaml exists
        for all 7 TS2Vec conditions, plus baseline_hmm.yaml and
        baseline_kmpca.yaml for the 2 external baselines.
        """
        expected = [
            "base.yaml",
            "experiment_1tf.yaml",
            "experiment_2tf.yaml",
            "experiment_3tf.yaml",
            "experiment_4tf.yaml",
            "experiment_bl_15m.yaml",
            "experiment_bl_4h.yaml",
            "experiment_bl_1d.yaml",
            "baseline_hmm.yaml",
            "baseline_kmpca.yaml",
        ]
        for filename in expected:
            assert (CONFIGS_DIR / filename).exists(), f"Missing: {filename}"
