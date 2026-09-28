"""Финальное обучение и проверка файла с предсказаниями."""

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import save_json
from config import ARTIFACT_DIR, SUBMISSION_DIR, ExperimentConfig
from experiments import add_frequency_features, fit_frequency_maps
from history_features import build_entity_history_features
from models import (
    fit_final_catboost,
    model_parameters,
    prepare_catboost,
)
from validation import (
    check_model_features,
    make_recency_weights,
)


LOGGER = logging.getLogger(__name__)


def validate_submission(submission, test, sample_submission):
    checks = {
        "columns": submission.columns.tolist() == ["cookie_id", "score"],
        "row_count": len(submission) == len(test),
        "unique_cookie_id": not submission["cookie_id"].duplicated().any(),
        "same_cookie_id": set(submission["cookie_id"]) == set(test["cookie_id"]),
        "sample_order": submission["cookie_id"].equals(
            sample_submission["cookie_id"]
        ),
        "numeric_score": pd.api.types.is_numeric_dtype(submission["score"]),
        "no_missing": not submission.isna().any().any(),
        "finite_score": bool(np.isfinite(submission["score"]).all()),
        "score_range": bool(submission["score"].between(0, 1).all()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Submission не прошёл проверки: " + ", ".join(failed))
    return checks


def load_or_build_final_history(
    config,
    train,
    test,
    events,
    artifact_dir=ARTIFACT_DIR,
):
    history_dir = Path(artifact_dir) / "feature_sets" / "final_history"
    entity_key = "_".join(config.history_entities)
    alpha_key = str(config.history_alpha).replace(".", "_")
    train_path = history_dir / f"{entity_key}_{alpha_key}_train.pkl"
    test_path = history_dir / f"{entity_key}_{alpha_key}_test.pkl"

    if train_path.exists() and test_path.exists():
        train_history = pd.read_pickle(train_path)
        test_history = pd.read_pickle(test_path)
    else:
        train_history, test_history = build_entity_history_features(
            train[["cookie_id", "window_start_ts", "target"]],
            test[["cookie_id"]],
            events,
            alpha=config.history_alpha,
            entity_prefixes=config.history_entities,
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        train_history.to_pickle(train_path)
        test_history.to_pickle(test_path)

    if set(train_history["cookie_id"]) != set(train["cookie_id"]):
        raise ValueError("Исторические признаки не покрывают весь train")
    if set(test_history["cookie_id"]) != set(test["cookie_id"]):
        raise ValueError("Исторические признаки не покрывают весь test")
    return train_history, test_history


def add_final_history(config, train_features, test_features, train, test, events):
    if not config.history_entities:
        return train_features, test_features
    train_history, test_history = load_or_build_final_history(
        config,
        train,
        test,
        events,
    )
    train_features = train_features.merge(
        train_history,
        on="cookie_id",
        validate="one_to_one",
    )
    test_features = test_features.merge(
        test_history,
        on="cookie_id",
        validate="one_to_one",
    )
    return train_features, test_features


def make_submission(
    config: ExperimentConfig,
    features,
    train,
    test,
    events,
    sample_submission,
    submission_group,
    submission_dir=SUBMISSION_DIR,
):
    """Обучает CatBoost на всём train и сохраняет отдельную версию ответа."""
    if config.model_name != "catboost":
        raise ValueError("Финальная сборка сейчас поддерживает только CatBoost")
    if test["window_start_ts"].min() < train["window_end_ts"].max():
        raise ValueError("Тест начинается до окончания обучающего периода")

    train_features = train[["cookie_id"]].merge(
        features,
        on="cookie_id",
        validate="one_to_one",
    )
    test_features = test[["cookie_id"]].merge(
        features,
        on="cookie_id",
        validate="one_to_one",
    )
    train_features, test_features = add_final_history(
        config,
        train_features,
        test_features,
        train,
        test,
        events,
    )

    frequency_maps = {}
    if config.use_frequency_features:
        frequency_maps = fit_frequency_maps(train_features)
        train_features = add_frequency_features(
            train_features,
            frequency_maps,
            style=config.frequency_feature_style,
        )
        test_features = add_frequency_features(
            test_features,
            frequency_maps,
            style=config.frequency_feature_style,
        )

    model_columns = [
        column for column in train_features.columns if column != "cookie_id"
    ]
    if model_columns != [
        column for column in test_features.columns if column != "cookie_id"
    ]:
        raise ValueError("Колонки train и test различаются")
    if (
        config.expected_feature_count is not None
        and len(model_columns) != config.expected_feature_count
    ):
        raise ValueError(
            "Состав финальной сборки изменился: "
            f"ожидалось {config.expected_feature_count} признаков, "
            f"получено {len(model_columns)}"
        )

    x_train = train_features[model_columns].copy()
    x_test = test_features[model_columns].copy()
    check_model_features(x_train, x_test)
    x_train, x_test, recipe = prepare_catboost(x_train, x_test)

    sample_weight = None
    if config.recency_half_life_days is not None:
        sample_weight = make_recency_weights(
            train["window_start_ts"],
            config.recency_half_life_days,
        )
    params = model_parameters("catboost", config.model_params)
    model, used_gpu = fit_final_catboost(
        x_train,
        train["target"].astype(int),
        recipe["category_columns"],
        params,
        sample_weight=sample_weight,
        compute_mode=config.compute_mode,
    )
    score = model.predict_proba(x_test)[:, 1]

    predictions = test[["cookie_id"]].copy()
    predictions["score"] = score
    submission = sample_submission[["cookie_id"]].merge(
        predictions,
        on="cookie_id",
        how="left",
        validate="one_to_one",
    )
    checks = validate_submission(submission, test, sample_submission)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(submission_dir) / submission_group / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    submission.to_csv(run_dir / "submission.csv", index=False)
    model.save_model(str(run_dir / "model.cbm"))
    save_json(model_columns, run_dir / "feature_columns.json")
    save_json(frequency_maps, run_dir / "frequency_maps.json")
    save_json(recipe, run_dir / "preprocessing.json")
    save_json(
        {
            "experiment": asdict(config),
            "model_params": params,
            "used_gpu": used_gpu,
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_count": len(model_columns),
            "checks": checks,
        },
        run_dir / "run_config.json",
    )
    LOGGER.info("Submission сохранён в %s", run_dir / "submission.csv")
    return run_dir, submission
