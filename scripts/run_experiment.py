"""Запуск одной или нескольких моделей на временной проверке."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from artifacts import configure_logging
from config import ExperimentConfig
from data import load_metadata, load_or_prepare_events
from experiments import run_time_cv
from feature_sets import FEATURE_SET_SPECS, build_feature_set
from models import SUPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", required=True, choices=FEATURE_SET_SPECS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()
    data = load_metadata()
    events = load_or_prepare_events(data.metadata)
    features, spec = build_feature_set(
        args.feature_set,
        data.metadata,
        data.train,
        events,
    )
    for model_name in args.models:
        config = ExperimentConfig(
            name=f"{model_name}_{args.feature_set}",
            feature_set=args.feature_set,
            model_name=model_name,
            history_entities=spec.history_entities,
            history_alpha=spec.history_alpha,
            recency_half_life_days=spec.recency_half_life_days,
        )
        summary = run_time_cv(
            config,
            features,
            data.train,
            events=events,
            force=args.force,
        )
        print(summary)


if __name__ == "__main__":
    main()
