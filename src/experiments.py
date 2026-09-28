"""Единый запуск моделей на временной кросс-валидации."""

import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from artifacts import load_json, make_run_dir, save_json
from config import ARTIFACT_DIR, TIME_FOLDS, ExperimentConfig
from history_features import build_entity_history_features
from models import fit_model, save_model
from validation import (
    calculate_metrics,
    make_recency_weights,
    split_by_time,
    summarize_fold_metrics,
)


LOGGER = logging.getLogger(__name__)

FREQUENCY_COLUMNS = (
    "ua_family",
    "main_category",
    "main_location",
    "main_seller_type",
    "category_platform",
    "category_seller",
)


def fit_frequency_maps(data):
    maps = {}
    for column in FREQUENCY_COLUMNS:
        if column not in data.columns:
            continue
        values = data[column].fillna("missing").astype(str)
        maps[column] = {
            str(key): float(value)
            for key, value in values.value_counts(normalize=True).items()
        }
    return maps


def add_frequency_features(data, maps, style="prefix"):
    if style not in {"prefix", "suffix"}:
        raise ValueError("Стиль частотных признаков должен быть prefix или suffix")
    result = data.copy()
    for column, value_map in maps.items():
        values = result[column].fillna("missing").astype(str)
        if style == "prefix":
            feature_name = f"frequency_{column}"
        else:
            feature_name = f"{column}_frequency"
        result[feature_name] = values.map(value_map).fillna(0.0)
    return result


def align_history(cookie_ids, history):
    return (
        pd.DataFrame({"cookie_id": cookie_ids})
        .merge(history, on="cookie_id", how="left", validate="one_to_one")
        .drop(columns="cookie_id")
        .reset_index(drop=True)
    )


def build_fold_history(config, fold_name, train_part, valid_part, events, run_dir):
    history_dir = run_dir / fold_name / "history"
    train_path = history_dir / "train.pkl"
    valid_path = history_dir / "valid.pkl"
    if train_path.exists() and valid_path.exists():
        train_history = pd.read_pickle(train_path)
        valid_history = pd.read_pickle(valid_path)
    else:
        train_history, valid_history = build_entity_history_features(
            train_part[["cookie_id", "window_start_ts", "target"]],
            valid_part[["cookie_id"]],
            events,
            alpha=config.history_alpha,
            entity_prefixes=config.history_entities,
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        train_history.to_pickle(train_path)
        valid_history.to_pickle(valid_path)

    if set(train_history["cookie_id"]) != set(train_part["cookie_id"]):
        raise ValueError(f"{fold_name}: история не покрывает train")
    if set(valid_history["cookie_id"]) != set(valid_part["cookie_id"]):
        raise ValueError(f"{fold_name}: история не покрывает valid")
    return train_history, valid_history


def combine_importance(tables):
    combined = None
    for fold_name, table in tables:
        current = table.set_index("feature")[["importance"]].rename(
            columns={"importance": fold_name}
        )
        combined = current if combined is None else combined.join(current, how="outer")
    combined = combined.fillna(0)
    fold_columns = combined.columns.tolist()
    combined["mean_importance"] = combined[fold_columns].mean(axis=1)
    combined["min_importance"] = combined[fold_columns].min(axis=1)
    return combined.sort_values("mean_importance", ascending=False).reset_index()


def run_time_cv(
    config: ExperimentConfig,
    features,
    train_metadata,
    events=None,
    artifact_dir=ARTIFACT_DIR,
    force=False,
):
    """Обучает одну конфигурацию на всех временных разбиениях."""
    config_dict = asdict(config)
    feature_columns = [column for column in features.columns if column != "cookie_id"]
    run_identity = {
        "config": config_dict,
        "feature_columns": feature_columns,
    }
    base_dir = Path(artifact_dir) / "experiments" / config.name
    run_dir = make_run_dir(base_dir, run_identity)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        LOGGER.info("%s: загружаю готовый запуск из %s", config.name, run_dir)
        return load_json(summary_path)

    if features["cookie_id"].duplicated().any():
        raise ValueError("В таблице признаков повторяется cookie_id")
    data = train_metadata[
        ["cookie_id", "window_start_ts", "window_end_ts", "target"]
    ].merge(
        features,
        on="cookie_id",
        how="left",
        validate="one_to_one",
    )
    if len(data) != len(train_metadata):
        raise ValueError("Признаки не соответствуют train")
    if config.history_entities and events is None:
        raise ValueError("Для исторических признаков нужны события")

    LOGGER.info(
        "%s: модель=%s, набор=%s, версия=%s, частоты=%s, устройство=%s, "
        "признаков до истории=%d",
        config.name,
        config.model_name,
        config.feature_set,
        config.feature_version or "не указана",
        config.frequency_feature_style,
        config.compute_mode,
        features.shape[1] - 1,
    )

    fold_rows = []
    prediction_tables = []
    importance_tables = []

    folds = [
        fold
        for fold in TIME_FOLDS
        if not config.fold_names or fold.name in config.fold_names
    ]
    if not folds:
        raise ValueError("Для эксперимента не выбрано ни одного разбиения")

    for fold in folds:
        train_part, valid_part = split_by_time(data, fold)
        drop_columns = ["cookie_id", "window_start_ts", "window_end_ts", "target"]
        x_train = train_part.drop(columns=drop_columns).reset_index(drop=True)
        x_valid = valid_part.drop(columns=drop_columns).reset_index(drop=True)
        y_train = train_part["target"].astype(int).reset_index(drop=True)
        y_valid = valid_part["target"].astype(int).reset_index(drop=True)

        if config.history_entities:
            train_history, valid_history = build_fold_history(
                config,
                fold.name,
                train_part,
                valid_part,
                events,
                run_dir,
            )
            x_train = pd.concat(
                [x_train, align_history(train_part["cookie_id"], train_history)],
                axis=1,
            )
            x_valid = pd.concat(
                [x_valid, align_history(valid_part["cookie_id"], valid_history)],
                axis=1,
            )

        frequency_maps = {}
        if config.use_frequency_features:
            frequency_maps = fit_frequency_maps(x_train)
            x_train = add_frequency_features(
                x_train,
                frequency_maps,
                style=config.frequency_feature_style,
            )
            x_valid = add_frequency_features(
                x_valid,
                frequency_maps,
                style=config.frequency_feature_style,
            )

        if (
            config.expected_feature_count is not None
            and x_train.shape[1] != config.expected_feature_count
        ):
            raise ValueError(
                f"{fold.name}: ожидалось {config.expected_feature_count} "
                f"признаков, получено {x_train.shape[1]}"
            )

        sample_weight = None
        if config.recency_half_life_days is not None:
            sample_weight = make_recency_weights(
                train_part["window_start_ts"].reset_index(drop=True),
                config.recency_half_life_days,
            )

        fold_dir = run_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(frequency_maps, fold_dir / "frequency_maps.json")
        save_json(x_train.columns.tolist(), fold_dir / "feature_columns.json")

        started = time.perf_counter()
        result = fit_model(
            config.model_name,
            x_train,
            y_train,
            x_valid,
            y_valid,
            params=config.model_params,
            sample_weight=sample_weight,
            compute_mode=config.compute_mode,
        )
        fit_seconds = time.perf_counter() - started
        metrics = calculate_metrics(y_valid, result.score)

        save_model(config.model_name, result.model, fold_dir / "model")
        save_json(result.preprocessing, fold_dir / "preprocessing.json")
        save_json(metrics, fold_dir / "metrics.json")
        predictions = pd.DataFrame(
            {
                "cookie_id": valid_part["cookie_id"].to_numpy(),
                "target": y_valid.to_numpy(),
                "score": result.score,
                "fold": fold.name,
            }
        )
        predictions.to_csv(fold_dir / "valid_predictions.csv", index=False)
        result.importance.to_csv(fold_dir / "feature_importance.csv", index=False)

        fold_rows.append(
            {
                "fold": fold.name,
                "train_rows": len(x_train),
                "valid_rows": len(x_valid),
                "feature_count": x_train.shape[1],
                "fit_seconds": fit_seconds,
                "used_gpu": result.used_gpu,
                **metrics,
            }
        )
        prediction_tables.append(predictions)
        importance_tables.append((fold.name, result.importance))
        LOGGER.info(
            "%s | %s | P@R70=%.4f | AP=%.4f | ROC-AUC=%.4f | %.1f с",
            config.name,
            fold.name,
            metrics["precision_at_recall_70"],
            metrics["average_precision"],
            metrics["roc_auc"],
            fit_seconds,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    importance = combine_importance(importance_tables)
    summary = {
        "experiment": config.name,
        "feature_set": config.feature_set,
        "feature_version": config.feature_version,
        "model": config.model_name,
        "frequency_feature_style": config.frequency_feature_style,
        "compute_mode": config.compute_mode,
        "run_hash": run_dir.name,
        "feature_count": int(fold_metrics["feature_count"].max()),
        "total_fit_seconds": float(fold_metrics["fit_seconds"].sum()),
        **summarize_fold_metrics(
            fold_metrics,
            predictions["target"],
            predictions["score"],
        ),
    }
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(run_dir / "oof_predictions.csv", index=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    save_json(summary, summary_path)
    save_json(run_identity, run_dir / "run_config.json")
    LOGGER.info(
        "%s завершён: mean P@R70=%.4f, last=%.4f",
        config.name,
        summary["mean_precision_at_recall_70"],
        summary["last_fold_precision_at_recall_70"],
    )
    return summary


def collect_experiment_summaries(artifact_dir=ARTIFACT_DIR):
    rows = []
    experiments_dir = Path(artifact_dir) / "experiments"
    for path in experiments_dir.glob("*/*/summary.json"):
        rows.append(load_json(path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["mean_precision_at_recall_70", "last_fold_precision_at_recall_70"],
        ascending=False,
    ).reset_index(drop=True)
