"""Единый интерфейс обучения и важности трёх моделей."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATBOOST_BASE_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
)
from validation import check_finite_numeric, check_model_features


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODELS = ("logistic_regression", "random_forest", "catboost")


@dataclass
class ModelResult:
    model: object
    score: np.ndarray
    importance: pd.DataFrame
    preprocessing: dict
    used_gpu: bool


def gpu_is_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def model_parameters(model_name, overrides=None):
    if model_name == "logistic_regression":
        params = LOGISTIC_REGRESSION_PARAMS.copy()
    elif model_name == "random_forest":
        params = RANDOM_FOREST_PARAMS.copy()
    elif model_name == "catboost":
        params = CATBOOST_BASE_PARAMS.copy()
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    params.update(overrides or {})
    return params


def make_sklearn_pipeline(model_name, train_data, params):
    numeric_columns = train_data.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    category_columns = [
        column for column in train_data.columns if column not in numeric_columns
    ]

    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
        model = LogisticRegression(**params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Для sklearn не поддерживается модель {model_name}")

    transformer = ColumnTransformer(
        [
            ("numbers", Pipeline(numeric_steps), numeric_columns),
            (
                "categories",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                category_columns,
            ),
        ]
    )
    return Pipeline([("prepare", transformer), ("model", model)])


def prepare_catboost(train_data, valid_data):
    train_ready = train_data.copy()
    valid_ready = valid_data.copy()
    numeric_columns = train_ready.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    category_columns = [
        column for column in train_ready.columns if column not in numeric_columns
    ]

    train_ready[numeric_columns] = train_ready[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    valid_ready[numeric_columns] = valid_ready[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    medians = train_ready[numeric_columns].median().fillna(0)
    train_ready[numeric_columns] = train_ready[numeric_columns].fillna(medians)
    valid_ready[numeric_columns] = valid_ready[numeric_columns].fillna(medians)

    for column in category_columns:
        train_ready[column] = train_ready[column].fillna("missing").astype(str)
        valid_ready[column] = valid_ready[column].fillna("missing").astype(str)

    check_finite_numeric(train_ready, "train")
    check_finite_numeric(valid_ready, "valid")
    recipe = {
        "numeric_columns": numeric_columns,
        "category_columns": category_columns,
        "numeric_medians": medians.to_dict(),
    }
    return train_ready, valid_ready, recipe


def fit_catboost(
    train_data,
    target,
    valid_data,
    valid_target,
    params,
    category_columns,
    sample_weight=None,
    compute_mode="auto",
):
    params = params.copy()
    if compute_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("compute_mode должен быть auto, cpu или gpu")
    use_gpu = compute_mode == "gpu" or (
        compute_mode == "auto" and gpu_is_available()
    )
    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})

    fit_params = {
        "cat_features": category_columns,
        "eval_set": (valid_data, valid_target),
        "use_best_model": False,
    }
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    model = CatBoostClassifier(**params)
    try:
        model.fit(train_data, target, **fit_params)
        return model, use_gpu
    except CatBoostError as error:
        message = str(error).lower()
        gpu_error = any(
            word in message for word in ("cuda", "gpu", "driver", "device")
        )
        if not use_gpu or not gpu_error:
            raise

        LOGGER.warning("GPU недоступен, повторяю обучение CatBoost на CPU")
        cpu_params = params.copy()
        cpu_params.pop("task_type", None)
        cpu_params.pop("devices", None)
        model = CatBoostClassifier(**cpu_params)
        model.fit(train_data, target, **fit_params)
        return model, False


def fit_final_catboost(
    train_data,
    target,
    category_columns,
    params,
    sample_weight=None,
    compute_mode="auto",
):
    params = params.copy()
    if compute_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("compute_mode должен быть auto, cpu или gpu")
    use_gpu = compute_mode == "gpu" or (
        compute_mode == "auto" and gpu_is_available()
    )
    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})
    fit_params = {"cat_features": category_columns}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    model = CatBoostClassifier(**params)
    try:
        model.fit(train_data, target, **fit_params)
        return model, use_gpu
    except CatBoostError as error:
        message = str(error).lower()
        gpu_error = any(
            word in message for word in ("cuda", "gpu", "driver", "device")
        )
        if not use_gpu or not gpu_error:
            raise
        LOGGER.warning("GPU недоступен, обучаю финальный CatBoost на CPU")
        params.pop("task_type", None)
        params.pop("devices", None)
        model = CatBoostClassifier(**params)
        model.fit(train_data, target, **fit_params)
        return model, False


def sklearn_importance(model):
    names = model.named_steps["prepare"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = estimator.feature_importances_
    return pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance",
        ascending=False,
    )


def catboost_importance(model):
    return pd.DataFrame(
        {
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)


def fit_model(
    model_name,
    train_data,
    target,
    valid_data,
    valid_target,
    params=None,
    sample_weight=None,
    compute_mode="auto",
):
    check_model_features(train_data, valid_data)
    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan)
    params = model_parameters(model_name, params)

    if model_name == "catboost":
        train_ready, valid_ready, recipe = prepare_catboost(
            train_data,
            valid_data,
        )
        model, used_gpu = fit_catboost(
            train_ready,
            target,
            valid_ready,
            valid_target,
            params,
            recipe["category_columns"],
            sample_weight=sample_weight,
            compute_mode=compute_mode,
        )
        score = model.predict_proba(valid_ready)[:, 1]
        importance = catboost_importance(model)
    else:
        model = make_sklearn_pipeline(model_name, train_data, params)
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        model.fit(train_data, target, **fit_params)
        score = model.predict_proba(valid_data)[:, 1]
        importance = sklearn_importance(model)
        recipe = {}
        used_gpu = False

    return ModelResult(model, score, importance, recipe, used_gpu)


def save_model(model_name, model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        model.save_model(str(path.with_suffix(".cbm")))
    else:
        joblib.dump(model, path.with_suffix(".joblib"))
