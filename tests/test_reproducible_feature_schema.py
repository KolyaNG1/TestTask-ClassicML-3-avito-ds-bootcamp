import pandas as pd

from experiments import add_frequency_features
from selected_features import (
    COMPACT_BEHAVIOR_COLUMNS_V1,
    DOMAIN_NUMERIC_COLUMNS,
    RANKED_NUMERIC_COLUMNS,
    STABLE_CATEGORICAL_COLUMNS,
)


def test_compact_schema_has_expected_groups():
    assert len(RANKED_NUMERIC_COLUMNS) == 40
    assert len(DOMAIN_NUMERIC_COLUMNS) == 12
    assert len(STABLE_CATEGORICAL_COLUMNS) == 8
    assert len(COMPACT_BEHAVIOR_COLUMNS_V1) == 60
    assert len(set(COMPACT_BEHAVIOR_COLUMNS_V1)) == 60


def test_frequency_feature_names_are_explicit():
    data = pd.DataFrame({"ua_family": ["chrome", "firefox"]})
    maps = {"ua_family": {"chrome": 0.5, "firefox": 0.5}}

    prefix = add_frequency_features(data, maps, style="prefix")
    suffix = add_frequency_features(data, maps, style="suffix")

    assert "frequency_ua_family" in prefix.columns
    assert "ua_family_frequency" in suffix.columns
