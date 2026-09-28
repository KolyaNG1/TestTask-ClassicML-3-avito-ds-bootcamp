"""Составы наборов признаков и их кеширование."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from artifacts import load_json, load_or_build_frame
from behavior_features import build_feature_sets as build_behavior_sets
from config import ARTIFACT_DIR, TIME_FOLDS
from history_features import (
    MissingnessFeatures,
    PointerDynamicsFeatures,
    SequenceHypothesisFeatures,
    TextHashFeatures,
)
from selected_features import COMPACT_BEHAVIOR_COLUMNS_V1


LOGGER = logging.getLogger(__name__)

BEHAVIOR_SET_NAMES = (
    "minimal_baseline",
    "basic_behavior",
    "temporal_behavior",
    "behavior_structure",
    "full_behavior",
    "extended_behavior",
)


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    base_name: str
    history_entities: tuple[str, ...] = ()
    history_alpha: float = 25.0
    recency_half_life_days: float | None = None


FEATURE_SET_SPECS = {
    name: FeatureSetSpec(name=name, base_name=name)
    for name in BEHAVIOR_SET_NAMES
}
FEATURE_SET_SPECS.update(
    {
        "compact_behavior": FeatureSetSpec(
            name="compact_behavior",
            base_name="compact_behavior",
        ),
        "rich_behavior": FeatureSetSpec(
            name="rich_behavior",
            base_name="rich_behavior",
        ),
        "behavior_with_entity_history": FeatureSetSpec(
            name="behavior_with_entity_history",
            base_name="rich_behavior",
            history_entities=(
                "item",
                "query",
                "user_agent",
                "category_location",
                "ua_category",
            ),
            recency_half_life_days=5.0,
        ),
        "behavior_with_non_item_history": FeatureSetSpec(
            name="behavior_with_non_item_history",
            base_name="rich_behavior",
            history_entities=(
                "query",
                "user_agent",
                "category_location",
                "ua_category",
            ),
            recency_half_life_days=5.0,
        ),
        "behavior_with_item_history": FeatureSetSpec(
            name="behavior_with_item_history",
            base_name="rich_behavior",
            history_entities=("item",),
            recency_half_life_days=5.0,
        ),
    }
)


def validate_feature_frame(frame, metadata):
    if "cookie_id" not in frame.columns:
        raise ValueError("В признаках нет cookie_id")
    if frame["cookie_id"].duplicated().any():
        raise ValueError("В признаках повторяется cookie_id")
    if set(frame["cookie_id"]) != set(metadata["cookie_id"]):
        raise ValueError("Признаки не покрывают все переданные куки")


def save_behavior_feature_catalog(frames, artifact_dir=ARTIFACT_DIR):
    """Сохраняет точный машинно-читаемый состав поведенческих наборов."""
    rows = []
    for set_name, frame in frames.items():
        for column in frame.columns:
            if column == "cookie_id":
                continue
            rows.append(
                {
                    "feature_set": set_name,
                    "feature": column,
                    "dtype": str(frame[column].dtype),
                }
            )
    catalog = pd.DataFrame(rows)
    path = Path(artifact_dir) / "feature_sets" / "behavior_feature_catalog.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    return catalog


def merge_feature_blocks(base, blocks):
    result = base.copy()
    for block in blocks:
        result = result.merge(
            block,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )
    return result


def load_behavior_feature_sets(
    metadata,
    train_metadata,
    events,
    artifact_dir=ARTIFACT_DIR,
):
    feature_dir = Path(artifact_dir) / "feature_sets" / "behavior"
    paths = {name: feature_dir / f"{name}.pkl" for name in BEHAVIOR_SET_NAMES}
    if all(path.exists() for path in paths.values()):
        frames = {name: pd.read_pickle(path) for name, path in paths.items()}
        for frame in frames.values():
            validate_feature_frame(frame, metadata)
        save_behavior_feature_catalog(frames, artifact_dir)
        LOGGER.info("Поведенческие наборы загружены из кеша")
        return frames

    first_validation_start = pd.Timestamp(TIME_FOLDS[0].valid_start)
    vocabulary_cookie_ids = train_metadata.loc[
        train_metadata["window_start_ts"] < first_validation_start,
        "cookie_id",
    ].tolist()
    if not vocabulary_cookie_ids:
        raise ValueError("Пустая выборка для словарей категориальных признаков")

    frames = build_behavior_sets(
        metadata,
        events,
        vocabulary_cookie_ids=vocabulary_cookie_ids,
        events_are_prepared=True,
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        validate_feature_frame(frame, metadata)
        frame.to_pickle(paths[name])
    save_behavior_feature_catalog(frames, artifact_dir)
    LOGGER.info("Поведенческие наборы сохранены в %s", feature_dir)
    return frames


def load_compact_behavior(
    behavior_sets,
    metadata,
    artifact_dir=ARTIFACT_DIR,
):
    columns_path = (
        Path(artifact_dir)
        / "feature_selection"
        / "compact_behavior_columns.json"
    )
    if not columns_path.exists():
        raise FileNotFoundError(
            "Не найден список compact_behavior_columns.json. "
            "Сначала запустите notebooks/catboost_selection.ipynb"
        )
    columns = load_json(columns_path)
    expected_columns = list(COMPACT_BEHAVIOR_COLUMNS_V1)
    if columns != expected_columns:
        raise ValueError(
            "Состав compact_behavior отличается от зафиксированной версии. "
            "Повторно запустите ноутбук отбора признаков."
        )
    extended = behavior_sets["extended_behavior"]
    missing = [column for column in columns if column not in extended.columns]
    if missing:
        raise ValueError(
            "В extended_behavior отсутствуют колонки: " + ", ".join(missing)
        )
    compact = extended[["cookie_id"] + columns].copy()
    validate_feature_frame(compact, metadata)
    return compact


def load_local_blocks(metadata, events, artifact_dir=ARTIFACT_DIR):
    block_dir = Path(artifact_dir) / "feature_sets" / "local_blocks"
    builders = {
        "sequence": lambda: SequenceHypothesisFeatures.build(metadata, events),
        "pointer_dynamics": lambda: PointerDynamicsFeatures.build(metadata, events),
        "missingness": lambda: MissingnessFeatures.build(metadata, events),
        "text_hash": lambda: TextHashFeatures.build(metadata, events),
    }
    blocks = {}
    for name, builder in builders.items():
        blocks[name] = load_or_build_frame(
            block_dir / f"{name}.pkl",
            builder=builder,
            validator=lambda frame: validate_feature_frame(frame, metadata),
            description=f"Блок признаков {name}",
        )
    return blocks


def build_feature_set(
    name,
    metadata,
    train_metadata,
    events,
    artifact_dir=ARTIFACT_DIR,
):
    if name not in FEATURE_SET_SPECS:
        raise ValueError(f"Неизвестный набор признаков: {name}")
    spec = FEATURE_SET_SPECS[name]
    behavior_sets = load_behavior_feature_sets(
        metadata,
        train_metadata,
        events,
        artifact_dir=artifact_dir,
    )
    if spec.base_name in behavior_sets:
        return behavior_sets[spec.base_name], spec

    compact = load_compact_behavior(
        behavior_sets,
        metadata,
        artifact_dir=artifact_dir,
    )
    if spec.base_name == "compact_behavior":
        return compact, spec

    local_blocks = load_local_blocks(metadata, events, artifact_dir=artifact_dir)
    rich = merge_feature_blocks(compact, list(local_blocks.values()))
    validate_feature_frame(rich, metadata)
    return rich, spec
