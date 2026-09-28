"""Общие настройки проекта.

В этом файле лежат только значения, которые должны быть одинаковыми во всех
ноутбуках и скриптах. В частности, проект использует один random seed.
"""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

RANDOM_SEED = 42
TARGET_RECALL = 0.70


@dataclass(frozen=True)
class TimeFold:
    name: str
    valid_start: str
    valid_end: str


TIME_FOLDS = (
    TimeFold("fold_01", "2026-04-13", "2026-04-15"),
    TimeFold("fold_02", "2026-04-15", "2026-04-17"),
    TimeFold("fold_03", "2026-04-17", "2026-04-20"),
)


LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 3000,
    "random_state": RANDOM_SEED,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

CATBOOST_BASE_PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_SEED,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}

CATBOOST_FINAL_PARAMS = {
    **CATBOOST_BASE_PARAMS,
    "iterations": 1200,
    "depth": 3,
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature_set: str
    model_name: str
    model_params: dict = field(default_factory=dict)
    fold_names: tuple[str, ...] = ()
    use_frequency_features: bool = True
    frequency_feature_style: str = "prefix"
    history_entities: tuple[str, ...] = ()
    history_alpha: float = 25.0
    recency_half_life_days: float | None = None
    expected_feature_count: int | None = None
    feature_version: str = ""
    compute_mode: str = "auto"
    random_seed: int = RANDOM_SEED
