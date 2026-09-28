"""Сохранение кешей, моделей и результатов экспериментов."""

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging(level=logging.INFO):
    """Включает единый короткий формат сообщений для всего проекта."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root_logger.setLevel(level)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def config_hash(data, length=10):
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def load_or_build_frame(path, builder, validator=None, description="таблица"):
    """Загружает таблицу из кеша или строит и сохраняет её."""
    path = Path(path)
    if path.exists():
        frame = pd.read_pickle(path)
        source = "кеш"
    else:
        started = time.perf_counter()
        frame = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
        source = f"расчёт за {time.perf_counter() - started:.1f} с"

    if validator is not None:
        validator(frame)

    LOGGER.info(
        "%s: %s, строк=%d, колонок=%d",
        description,
        source,
        len(frame),
        frame.shape[1],
    )
    return frame


def make_run_dir(base_dir, config):
    """Создаёт устойчивый путь запуска на основе его конфигурации."""
    run_name = config_hash(config)
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, run_dir / "config.json")
    return run_dir
