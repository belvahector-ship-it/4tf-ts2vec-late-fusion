"""
src/utils/paths.py

Centralized path constants for the entire pipeline (M0 — Project Bootstrap).

Purpose
-------
Every module that reads or writes a file on disk imports its paths from
here instead of hardcoding relative paths. This keeps the repository
runnable from any working directory (e.g. `python scripts/run_m1.py`
from the repo root, or from inside `scripts/`) and gives a single place
to change the on-disk layout if it ever needs to move.

Design notes
------------
- `get_project_root()` locates the repository root by walking up from
  this file's location, not from `os.getcwd()`. This makes path
  resolution independent of where the calling script was launched from.
- All directory constants are created (if missing) the first time this
  module is imported, EXCEPT directories that are expected to be
  populated by download/training steps (data/raw, checkpoints,
  experiments) — those are created on demand by the modules that write
  into them, so an empty repo clone doesn't silently create misleading
  empty folders before M1/M8 have actually run.
- Per IMP-01 M0 Definition of Done, `data/`, `checkpoints/`, `outputs/`,
  and `experiments/` are excluded from version control (.gitignore) but
  the folder skeleton itself (with `.gitkeep`) IS versioned so a fresh
  clone has the expected directory shape.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the absolute path to the repository root.

    Resolution strategy: this file lives at `<root>/src/utils/paths.py`,
    so the root is three levels up from this file's resolved location.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


# --- Top-level directories -------------------------------------------------

PROJECT_ROOT: Path = get_project_root()

SRC_DIR: Path = PROJECT_ROOT / "src"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUTS_FINAL_DIR: Path = OUTPUTS_DIR / "final"
OUTPUTS_FIGURES_DIR: Path = OUTPUTS_FINAL_DIR / "figures"
OUTPUTS_TABLES_DIR: Path = OUTPUTS_FINAL_DIR / "tables"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

# --- Frequently used specific files -----------------------------------------

BASE_CONFIG_PATH: Path = CONFIGS_DIR / "base.yaml"
RAW_MANIFEST_PATH: Path = DATA_RAW_DIR / "manifest.json"
EXPERIMENT_REGISTRY_PATH: Path = EXPERIMENTS_DIR / "registry.json"

# --- Experimental condition labels (DS-03 §4, IMP-01 v1.1) -----------------
# 7 unique TS2Vec conditions: 4 primary cumulative + 3 secondary baselines.
# 1TF and BL-1h refer to the identical condition/branch and are NOT double
# counted (PROPOSAL §3.6, DS-03 §4).
TS2VEC_CONDITIONS: tuple[str, ...] = (
    "1TF",
    "2TF",
    "3TF",
    "4TF",
    "BL-15m",
    "BL-4h",
    "BL-1d",
)

# 2 external baselines, run under the identical five-seed protocol.
EXTERNAL_BASELINES: tuple[str, ...] = ("HMM", "KM-PCA")

# All 4 timeframes used across the branch encoders.
TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1d")

# 5 random seeds applied identically to every source of randomness
# (Python, NumPy, PyTorch, CUDA) per DS-03 Table 3.11 / IMP-01.
RANDOM_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)


def ensure_dir(path: Path) -> Path:
    """
    Create `path` (and parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory to ensure exists.

    Returns
    -------
    Path
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_dir(exp_id: str) -> Path:
    """
    Return (and ensure) the directory for a single experiment run.

    Parameters
    ----------
    exp_id : str
        Unique, human-readable experiment ID, e.g. "20260601_1TF_seed42".

    Returns
    -------
    Path
        `experiments/{exp_id}/`, created if missing.
    """
    return ensure_dir(EXPERIMENTS_DIR / exp_id)


def get_branch_checkpoint_dir(timeframe: str) -> Path:
    """
    Return (and ensure) the checkpoint directory for one branch encoder.

    Parameters
    ----------
    timeframe : str
        One of TIMEFRAMES ("15m", "1h", "4h", "1d").

    Returns
    -------
    Path
        `checkpoints/branch_{timeframe}/`, created if missing.

    Raises
    ------
    ValueError
        If `timeframe` is not one of the four recognized timeframes.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. Expected one of {TIMEFRAMES}."
        )
    return ensure_dir(CHECKPOINTS_DIR / f"branch_{timeframe}")


# Directories that are safe to create eagerly at import time: they hold
# versioned content (configs, docs) or are pure organizational skeletons
# that tests/M0 validation expect to exist immediately after clone.
for _dir in (
    CONFIGS_DIR,
    DOCS_DIR,
    DOCS_DIR / "checkpoints",
    LOGS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
