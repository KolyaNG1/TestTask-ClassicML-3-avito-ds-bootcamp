import numpy as np

from metric import precision_at_recall, recall_at_fpr


def test_official_example():
    target = [1, 0, 1, 1, 1]
    score = [5, 4, 3, 2, 1]
    assert precision_at_recall(target, score) == 0.8


def test_equal_scores_are_one_group():
    assert precision_at_recall([1, 1, 0, 0], [0.5] * 4) == 0.5
    assert precision_at_recall([0, 0, 1, 1], [0.5] * 4) == 0.5


def test_recall_at_fpr_does_not_split_equal_scores():
    target = np.r_[np.ones(10), np.zeros(90)]
    score = np.full(100, 0.5)
    assert recall_at_fpr(target, score) == 0.0
