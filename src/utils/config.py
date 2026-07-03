"""
src/utils/config.py

YAML configuration loader with schema validation (M0 — Project Bootstrap).

Purpose
-------
All controlled variables from DS-03 §3 (Table 3.11) live in
`configs/base.yaml`. No module in this repository may hardcode a
scientific hyperparameter — every numeric constant that affects an
experimental outcome must be read from a config file loaded through
this module. `load_config` fails loudly and informatively if a required
field is missing, so a broken or incomplete config cannot silently
produce a scientifically invalid run.

This module deliberately has zero dependency on any other `src/` module
except `paths.py`, since it is a foundation that everything else in the
pipeline depends on (IMP-01 M0, Implementation Order 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# The set of fields every experiment config (base + per-condition) must
# contain. This mirrors DS-03 Table 3.11 "Controlled Variables". Nested
# fields are expressed with dotted paths, e.g. "training.optimizer".
REQUIRED_BASE_FIELDS: tuple[str, ...] = (
    "encoder.architecture",
    "encoder.input_dim",
    "encoder.hidden_dim",
    "encoder.output_dim",
    "encoder.depth",
    "encoder.kernel_size",
    "encoder.mask_ratio",
    "fusion.output_dim",
    "fusion.projection_seed",
    "window.size",
    "window.stride",
    "window.n_features",
    "training.optimizer",
    "training.learning_rate",
    "training.weight_decay",
    "training.batch_size",
    "training.max_epochs",
    "training.early_stopping_patience",
    "clustering.algorithm",
    "clustering.distance_metric",
    "clustering.min_cluster_size_grid",
    "clustering.min_samples_grid",
    "clustering.max_clusters",
    "dataset.symbol",
    "dataset.exchange",
    "dataset.start_date",
    "dataset.end_date",
    "seeds",
)


class ConfigValidationError(ValueError):
    """Raised when a loaded config is missing required fields or is malformed."""


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    """
    Retrieve a value from a nested dict using a dotted path.

    Parameters
    ----------
    d : dict
        The dictionary to search.
    dotted_key : str
        A dotted path, e.g. "encoder.output_dim".

    Returns
    -------
    Any
        The value at that path.

    Raises
    ------
    KeyError
        If any segment of the path is missing.
    """
    parts = dotted_key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def validate_schema(
    config: dict[str, Any],
    required_fields: tuple[str, ...] = REQUIRED_BASE_FIELDS,
) -> None:
    """
    Validate that `config` contains every field in `required_fields`.

    Parameters
    ----------
    config : dict
        The loaded configuration dictionary.
    required_fields : tuple[str, ...]
        Dotted-path field names that must be present.

    Raises
    ------
    ConfigValidationError
        If one or more required fields are missing. The error message
        lists every missing field at once (not just the first one
        found), so a researcher can fix the config in one pass.
    """
    missing: list[str] = []
    for field in required_fields:
        try:
            _get_nested(config, field)
        except KeyError:
            missing.append(field)

    if missing:
        missing_list = "\n".join(f"  - {f}" for f in missing)
        raise ConfigValidationError(
            "Config validation failed: the following required fields "
            f"are missing:\n{missing_list}\n\n"
            "See DS-03 Table 3.11 (Controlled Variables) and "
            "REQUIRED_BASE_FIELDS in src/utils/config.py for the full "
            "list of required fields."
        )


def load_config(
    path: Path,
    required_fields: tuple[str, ...] = REQUIRED_BASE_FIELDS,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the YAML config file.
    required_fields : tuple[str, ...], optional
        Dotted-path fields that must be present. Defaults to
        REQUIRED_BASE_FIELDS (the full DS-03 controlled-variables set).
        Pass a smaller tuple for configs that intentionally do not
        carry the full schema (e.g. `configs/baseline_hmm.yaml`, which
        does not need `encoder.*` or `window.*` fields).
    validate : bool, optional
        If False, skip schema validation entirely (default True).
        Only intended for loading partial/auxiliary configs that are
        merged into a base config elsewhere.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ConfigValidationError
        If required fields are missing (and `validate=True`).
    yaml.YAMLError
        If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Check that the path is correct and that you are running "
            "from the repository root (or using an absolute path)."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigValidationError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"Config file must parse to a YAML mapping (dict) at the "
            f"top level, got {type(config).__name__}: {path}"
        )

    if validate:
        validate_schema(config, required_fields=required_fields)

    return config


def load_condition_config(condition_config_path: Path, base_config_path: Path) -> dict[str, Any]:
    """
    Load a per-condition config and merge it over the base config.

    Per-condition configs (e.g. `configs/experiment_1tf.yaml`) only need
    to specify what differs from the base (per INV-001, only the set of
    active temporal resolutions may differ between conditions). This
    function loads `base.yaml` first, then applies the condition-specific
    overrides (typically just `condition.name` and `condition.active_timeframes`),
    and validates the merged result against the full required schema.

    Parameters
    ----------
    condition_config_path : Path
        Path to the per-condition YAML file.
    base_config_path : Path
        Path to `configs/base.yaml`.

    Returns
    -------
    dict
        The merged, fully validated configuration.
    """
    base = load_config(base_config_path, validate=True)
    override = load_config(
        condition_config_path,
        required_fields=("condition.name", "condition.active_timeframes"),
        validate=True,
    )

    merged = _deep_merge(base, override)
    validate_schema(merged, required_fields=REQUIRED_BASE_FIELDS)
    validate_schema(
        merged, required_fields=("condition.name", "condition.active_timeframes")
    )
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge `override` into a copy of `base`.

    Parameters
    ----------
    base : dict
        Base dictionary (not mutated).
    override : dict
        Dictionary whose values take precedence.

    Returns
    -------
    dict
        A new merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
