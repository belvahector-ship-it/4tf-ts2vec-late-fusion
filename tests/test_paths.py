"""
tests/test_paths.py

Unit tests for src/utils/paths.py (M0 — Project Bootstrap).

Covers IMP-01 Milestone 1 Completion Criteria: "Directory structure
exists" and verifies the condition/timeframe/seed constants match
DS-03 exactly (source of the M0->M1..M15 IMP-01 v1.1 correction).
"""

from __future__ import annotations

from src.utils.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    EXTERNAL_BASELINES,
    PROJECT_ROOT,
    RANDOM_SEEDS,
    TIMEFRAMES,
    TS2VEC_CONDITIONS,
    get_branch_checkpoint_dir,
    get_experiment_dir,
    get_project_root,
)


class TestProjectRoot:
    def test_project_root_exists(self) -> None:
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_expected_top_level_dirs(self) -> None:
        expected = {"src", "configs", "tests", "scripts", "docs"}
        actual = {p.name for p in PROJECT_ROOT.iterdir() if p.is_dir()}
        assert expected.issubset(actual)

    def test_get_project_root_is_consistent(self) -> None:
        assert get_project_root() == PROJECT_ROOT


class TestDirectoryStructure:
    def test_configs_dir_exists(self) -> None:
        assert CONFIGS_DIR.exists()

    def test_data_subdirs_exist(self) -> None:
        assert (DATA_DIR / "raw").exists()
        assert (DATA_DIR / "interim").exists()
        assert (DATA_DIR / "processed").exists()


class TestExperimentalConstants:
    """
    These constants are the direct fix for the IMP-01 v1.0 -> v1.1
    inconsistency: 7 TS2Vec conditions (not 8), 2 external baselines,
    45 total runs (not 40).
    """

    def test_seven_unique_ts2vec_conditions(self) -> None:
        assert len(TS2VEC_CONDITIONS) == 7
        assert len(set(TS2VEC_CONDITIONS)) == 7  # all unique

    def test_ts2vec_condition_names_match_ds03(self) -> None:
        expected = {"1TF", "2TF", "3TF", "4TF", "BL-15m", "BL-4h", "BL-1d"}
        assert set(TS2VEC_CONDITIONS) == expected

    def test_two_external_baselines(self) -> None:
        assert len(EXTERNAL_BASELINES) == 2
        assert set(EXTERNAL_BASELINES) == {"HMM", "KM-PCA"}

    def test_four_timeframes(self) -> None:
        assert len(TIMEFRAMES) == 4
        assert set(TIMEFRAMES) == {"15m", "1h", "4h", "1d"}

    def test_five_random_seeds_match_ds03_table_3_11(self) -> None:
        assert RANDOM_SEEDS == (42, 123, 456, 789, 1024)

    def test_total_run_count_is_45(self) -> None:
        """
        DS-03 §6 / IMP-01 v1.1: 7 TS2Vec conditions x 5 seeds = 35,
        plus 2 external baselines x 5 seeds = 10. Total = 45.
        """
        ts2vec_runs = len(TS2VEC_CONDITIONS) * len(RANDOM_SEEDS)
        external_runs = len(EXTERNAL_BASELINES) * len(RANDOM_SEEDS)
        assert ts2vec_runs == 35
        assert external_runs == 10
        assert ts2vec_runs + external_runs == 45

    def test_branch_checkpoint_count_is_20(self) -> None:
        """
        4 timeframes x 5 seeds = 20 branch checkpoints, independent of
        the 7-condition count (secondary baselines reuse the same 4
        branch encoders — IMP-01 v1.1 changelog note).
        """
        assert len(TIMEFRAMES) * len(RANDOM_SEEDS) == 20


class TestPathHelpers:
    def test_get_experiment_dir_creates_directory(self, tmp_path, monkeypatch) -> None:
        import src.utils.paths as paths_module

        monkeypatch.setattr(paths_module, "EXPERIMENTS_DIR", tmp_path)
        result = get_experiment_dir("20260701_1TF_seed42")
        assert result.exists()
        assert result == tmp_path / "20260701_1TF_seed42"

    def test_get_branch_checkpoint_dir_valid_timeframe(self, tmp_path, monkeypatch) -> None:
        import src.utils.paths as paths_module

        monkeypatch.setattr(paths_module, "CHECKPOINTS_DIR", tmp_path)
        result = get_branch_checkpoint_dir("1h")
        assert result.exists()
        assert result.name == "branch_1h"

    def test_get_branch_checkpoint_dir_invalid_timeframe_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_branch_checkpoint_dir("30m")
