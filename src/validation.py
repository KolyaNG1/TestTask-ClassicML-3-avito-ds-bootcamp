"""Временные разбиения, проверки утечек и расчёт метрик."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import TARGET_RECALL, TimeFold
from metric import precision_at_recall, recall_at_fpr


LOGGER = logging.getLogger(__name__)

FORBIDDEN_COLUMNS = {
    "cookie_id",
    "target",
    "eid",
    "event_ts",
    "window_start_ts",
    "window_end_ts",
    "cookie_created_at",
}


def split_by_time(data, fold: TimeFold):
    dates = pd.to_datetime(data["window_start_ts"])
    valid_start = pd.Timestamp(fold.valid_start)
    valid_end = pd.Timestamp(fold.valid_end)
    train_part = data.loc[dates < valid_start].copy()
    valid_part = data.loc[(dates >= valid_start) & (dates < valid_end)].copy()
    validate_time_split(train_part, valid_part, fold.name)
    return train_part, valid_part


def validate_time_split(train_part, valid_part, fold_name="fold"):
    if train_part.empty or valid_part.empty:
        raise ValueError(f"{fold_name}: пустая обучающая или проверочная часть")
    train_end = pd.to_datetime(train_part["window_end_ts"]).max()
    valid_start = pd.to_datetime(valid_part["window_start_ts"]).min()
    if train_end > valid_start:
        raise ValueError(f"{fold_name}: обучающие окна пересекаются с проверкой")
    if set(train_part["cookie_id"]) & set(valid_part["cookie_id"]):
        raise ValueError(f"{fold_name}: cookie_id пересекаются")
    LOGGER.info(
        "%s: train=%d, valid=%d, valid_bots=%d",
        fold_name,
        len(train_part),
        len(valid_part),
        int(valid_part["target"].sum()),
    )


def forbidden_model_columns(columns):
    return [
        column
        for column in columns
        if column in FORBIDDEN_COLUMNS or column.endswith("_id")
    ]


def check_model_features(train_data, valid_data):
    if train_data.columns.tolist() != valid_data.columns.tolist():
        raise ValueError("Колонки обучающей и проверочной частей различаются")
    forbidden = forbidden_model_columns(train_data.columns)
    if forbidden:
        raise ValueError(
            "В модель попали идентификаторы или даты: " + ", ".join(forbidden)
        )


def check_finite_numeric(data, name="data"):
    numeric = data.select_dtypes(include=["number", "bool"])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"В числовых признаках {name} есть пропуски или бесконечность")


def make_recency_weights(dates, half_life_days):
    dates = pd.to_datetime(dates)
    age_days = (dates.max() - dates).dt.total_seconds() / 86400
    weights = np.power(0.5, age_days / half_life_days)
    return weights / weights.mean()


def calculate_metrics(target, score):
    return {
        "precision_at_recall_70": precision_at_recall(
            target,
            score,
            recall=TARGET_RECALL,
        ),
        "average_precision": average_precision_score(target, score),
        "roc_auc": roc_auc_score(target, score),
        "recall_at_fpr_01": recall_at_fpr(target, score, fpr=0.01),
        "unique_scores": int(pd.Series(score).nunique()),
    }


def summarize_fold_metrics(fold_metrics, pooled_target, pooled_score):
    pooled = calculate_metrics(pooled_target, pooled_score)
    return {
        "mean_precision_at_recall_70": float(
            fold_metrics["precision_at_recall_70"].mean()
        ),
        "std_precision_at_recall_70": float(
            fold_metrics["precision_at_recall_70"].std(ddof=0)
        ),
        "last_fold_precision_at_recall_70": float(
            fold_metrics.iloc[-1]["precision_at_recall_70"]
        ),
        "mean_average_precision": float(fold_metrics["average_precision"].mean()),
        "mean_roc_auc": float(fold_metrics["roc_auc"].mean()),
        "pooled_precision_at_recall_70": pooled["precision_at_recall_70"],
        "pooled_average_precision": pooled["average_precision"],
        "pooled_roc_auc": pooled["roc_auc"],
    }
