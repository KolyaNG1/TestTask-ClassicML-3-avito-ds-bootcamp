"""Подготовка событий и поведенческих наборов признаков."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from artifacts import configure_logging
from data import load_metadata, load_or_prepare_events
from feature_sets import load_behavior_feature_sets


def main():
    configure_logging()
    data = load_metadata()
    events = load_or_prepare_events(data.metadata)
    feature_sets = load_behavior_feature_sets(
        data.metadata,
        data.train,
        events,
    )
    for name, frame in feature_sets.items():
        print(f"{name}: {frame.shape[1] - 1} признаков")


if __name__ == "__main__":
    main()
