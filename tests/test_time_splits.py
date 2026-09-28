import pandas as pd

from config import TimeFold
from validation import split_by_time


def test_time_split_uses_only_past_windows():
    data = pd.DataFrame(
        {
            "cookie_id": ["a", "b", "c"],
            "window_start_ts": pd.to_datetime(
                ["2026-04-10", "2026-04-12", "2026-04-13"]
            ),
            "window_end_ts": pd.to_datetime(
                ["2026-04-11", "2026-04-13", "2026-04-14"]
            ),
            "target": [0, 1, 0],
        }
    )
    fold = TimeFold("test_fold", "2026-04-13", "2026-04-14")
    train, valid = split_by_time(data, fold)
    assert train["cookie_id"].tolist() == ["a", "b"]
    assert valid["cookie_id"].tolist() == ["c"]
    assert train["window_end_ts"].max() <= valid["window_start_ts"].min()
