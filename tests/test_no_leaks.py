import pandas as pd
import pytest

from validation import check_model_features, forbidden_model_columns


def test_raw_identifiers_are_forbidden():
    columns = ["event_count", "item_id", "cookie_id", "item_history_rate_mean"]
    assert forbidden_model_columns(columns) == ["item_id", "cookie_id"]


def test_model_check_rejects_identifier():
    train = pd.DataFrame({"event_count": [1], "item_id": [10]})
    valid = pd.DataFrame({"event_count": [2], "item_id": [11]})
    with pytest.raises(ValueError, match="идентификаторы"):
        check_model_features(train, valid)
