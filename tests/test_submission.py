import pandas as pd

from submission import validate_submission


def test_valid_submission_passes():
    test = pd.DataFrame({"cookie_id": ["a", "b"]})
    sample = pd.DataFrame({"cookie_id": ["b", "a"], "score": [0.0, 0.0]})
    submission = pd.DataFrame(
        {"cookie_id": ["b", "a"], "score": [0.2, 0.8]}
    )
    checks = validate_submission(submission, test, sample)
    assert all(checks.values())
