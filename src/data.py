"""Загрузка данных и подготовка событий внутри заданных окон."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from artifacts import load_or_build_frame
from config import ARTIFACT_DIR, DATA_DIR


LOGGER = logging.getLogger(__name__)

DATE_COLUMNS = ("cookie_created_at", "window_start_ts", "window_end_ts")
EVENT_TYPES = (
    "search_results_view",
    "item_view",
    "photo_swipe",
    "seller_page_view",
    "contact_phone_show",
    "contact_chat_open",
    "contact_message_sent",
    "favorite_add",
    "login",
)
RAW_EVENT_COLUMNS = (
    "cookie_id",
    "event_ts",
    "eid",
    "event_name",
    "platform",
    "user_agent",
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
)


@dataclass(frozen=True)
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame

    @property
    def metadata(self):
        return pd.concat(
            [self.train.drop(columns="target"), self.test],
            ignore_index=True,
        )


def resolve_data_paths(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.exists():
        sample_path = data_dir.parent / "sample_submission.csv"

    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample_submission": sample_path,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены файлы: " + ", ".join(missing))
    return paths


def validate_metadata(train, test):
    required = {"cookie_id", *DATE_COLUMNS}
    if not required.issubset(train.columns) or "target" not in train.columns:
        raise ValueError("В train отсутствуют обязательные колонки")
    if not required.issubset(test.columns):
        raise ValueError("В test отсутствуют обязательные колонки")
    if train["cookie_id"].duplicated().any():
        raise ValueError("В train повторяется cookie_id")
    if test["cookie_id"].duplicated().any():
        raise ValueError("В test повторяется cookie_id")
    if set(train["cookie_id"]) & set(test["cookie_id"]):
        raise ValueError("cookie_id пересекаются между train и test")
    if not train["target"].dropna().isin([0, 1]).all():
        raise ValueError("target должен содержать только 0 и 1")
    if test["window_start_ts"].min() < train["window_end_ts"].max():
        raise ValueError("Тестовый период начинается до окончания train")


def load_metadata(data_dir=DATA_DIR):
    paths = resolve_data_paths(data_dir)
    train = pd.read_csv(paths["train"], parse_dates=list(DATE_COLUMNS))
    test = pd.read_csv(paths["test"], parse_dates=list(DATE_COLUMNS))
    sample_submission = pd.read_csv(paths["sample_submission"])
    validate_metadata(train, test)
    LOGGER.info(
        "Метаданные загружены: train=%d, test=%d, ботов=%d",
        len(train),
        len(test),
        int(train["target"].sum()),
    )
    return DataBundle(train, test, sample_submission)


def find_events_path(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    for name in ("events.csv", "events.csv.gz"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"В {data_dir} не найден events.csv или events.csv.gz")


def load_raw_events(data_dir=DATA_DIR):
    path = find_events_path(data_dir)
    LOGGER.info("Чтение событий из %s", path.name)
    return pd.read_csv(path, parse_dates=["event_ts"])


def prepare_events(metadata, events):
    """Оставляет разрешённые события внутри окна каждой куки."""
    if metadata["cookie_id"].duplicated().any():
        raise ValueError("В метаданных повторяется cookie_id")
    missing = set(RAW_EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(
            "В событиях отсутствуют колонки: " + ", ".join(sorted(missing))
        )

    windows = metadata[["cookie_id", "window_start_ts", "window_end_ts"]].copy()
    windows["window_start_ts"] = pd.to_datetime(windows["window_start_ts"])
    windows["window_end_ts"] = pd.to_datetime(windows["window_end_ts"])

    prepared = events[list(RAW_EVENT_COLUMNS)].copy()
    prepared["event_ts"] = pd.to_datetime(prepared["event_ts"])
    prepared = prepared.merge(
        windows,
        on="cookie_id",
        how="inner",
        validate="many_to_one",
    )
    inside = (
        (prepared["event_ts"] >= prepared["window_start_ts"])
        & (prepared["event_ts"] < prepared["window_end_ts"])
    )
    prepared = prepared.loc[inside]
    prepared = prepared.loc[prepared["event_name"].isin(EVENT_TYPES)].copy()
    prepared["platform_norm"] = (
        prepared["platform"].str.strip().str.lower().replace({"iphone": "ios"})
    )
    prepared["extra_duplicate"] = prepared.duplicated(list(RAW_EVENT_COLUMNS))
    prepared = prepared.sort_values(
        ["cookie_id", "event_ts"],
        kind="mergesort",
    ).reset_index(drop=True)

    validate_prepared_events(metadata, prepared)
    LOGGER.info(
        "События подготовлены: строк=%d, кук=%d, лишних дублей=%d",
        len(prepared),
        prepared["cookie_id"].nunique(),
        int(prepared["extra_duplicate"].sum()),
    )
    return prepared


def validate_prepared_events(metadata, events):
    if events["event_name"].eq("captcha_shown").any():
        raise ValueError("В подготовленных событиях остался captcha_shown")
    inside = (
        (events["event_ts"] >= events["window_start_ts"])
        & (events["event_ts"] < events["window_end_ts"])
    )
    if not inside.all():
        raise ValueError("Найдены события за границами окна")
    unknown = set(events["cookie_id"]) - set(metadata["cookie_id"])
    if unknown:
        raise ValueError("В подготовленных событиях есть неизвестные cookie_id")
    if not events[["cookie_id", "event_ts"]].equals(
        events[["cookie_id", "event_ts"]].sort_values(
            ["cookie_id", "event_ts"],
            kind="mergesort",
        ).reset_index(drop=True)
    ):
        raise ValueError("События не отсортированы по cookie_id и event_ts")


def load_or_prepare_events(
    metadata,
    data_dir=DATA_DIR,
    artifact_dir=ARTIFACT_DIR,
):
    cache_path = Path(artifact_dir) / "prepared" / "events_in_window.pkl"
    return load_or_build_frame(
        cache_path,
        builder=lambda: prepare_events(metadata, load_raw_events(data_dir)),
        validator=lambda frame: validate_prepared_events(metadata, frame),
        description="События внутри окон",
    )
