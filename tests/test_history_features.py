import pandas as pd

from history_features import build_entity_history_features


def test_item_history_uses_only_previous_days():
    train = pd.DataFrame(
        {
            "cookie_id": ["day_1", "day_2_a", "day_2_b"],
            "window_start_ts": pd.to_datetime(
                ["2026-04-01", "2026-04-02", "2026-04-02"]
            ),
            "target": [1, 0, 1],
        }
    )
    valid = pd.DataFrame({"cookie_id": ["valid"]})
    events = pd.DataFrame(
        {
            "cookie_id": ["day_1", "day_2_a", "day_2_b", "valid"],
            "item_id": [100, 100, 100, 100],
            "search_query": ["q", "q", "q", "q"],
            "user_agent": ["ua", "ua", "ua", "ua"],
            "item_category": ["cat", "cat", "cat", "cat"],
            "item_location": ["loc", "loc", "loc", "loc"],
        }
    )

    train_history, valid_history = build_entity_history_features(
        train,
        valid,
        events,
        alpha=25.0,
        entity_prefixes=("item",),
    )
    history = train_history.set_index("cookie_id")

    assert history.loc["day_1", "item_history_count_max"] == 0
    assert history.loc["day_2_a", "item_history_count_max"] == 1
    assert history.loc["day_2_b", "item_history_count_max"] == 1
    assert valid_history.loc[0, "item_history_count_max"] == 3
    assert not any(column == "item_id" for column in train_history.columns)
