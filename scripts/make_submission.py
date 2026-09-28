"""Финальное обучение CatBoost и создание submission.csv."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from artifacts import configure_logging
from config import CATBOOST_FINAL_PARAMS, ExperimentConfig
from data import load_metadata, load_or_prepare_events
from feature_sets import FEATURE_SET_SPECS, build_feature_set
from submission import make_submission


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        default="behavior_with_item_history",
        choices=FEATURE_SET_SPECS,
    )
    parser.add_argument("--group", default="item_history")
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
    config = ExperimentConfig(
        name=f"final_{args.feature_set}",
        feature_set=args.feature_set,
        model_name="catboost",
        model_params=CATBOOST_FINAL_PARAMS,
        history_entities=spec.history_entities,
        history_alpha=spec.history_alpha,
        recency_half_life_days=spec.recency_half_life_days,
        expected_feature_count=(
            719 if args.feature_set == "behavior_with_item_history" else None
        ),
    )
    run_dir, _ = make_submission(
        config,
        features,
        data.train,
        data.test,
        events,
        data.sample_submission,
        submission_group=args.group,
    )
    print("Результат сохранён:", run_dir)


if __name__ == "__main__":
    main()
