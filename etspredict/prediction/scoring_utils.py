#!/usr/bin/env python3

"""
Leonard Sasse
10/10/2021
"""

import numpy as np


def avg_cv_scores(scores):
    """
    Function to average cross validation scores first across folds, then across
    repeats
    scores: pd.DataFrame
        each column should correspond to a fold, each row should correspond
        to a repeat
    returns
    ------
    mean cv score
    """
    rows, _ = scores.shape

    if rows == 1:
        return np.mean(scores.values), np.std(scores.values)
    else:
        avg_across_folds = scores.mean(axis=1)
        return avg_across_folds.mean(), avg_across_folds.std()


def compare_propabilites(p_left, p_rope, p_right):
    if (p_rope > p_left) and (p_rope > p_right):
        return "="
    elif (p_left > p_rope) and (p_left > p_right):
        return ">"
    elif (p_right > p_rope) and (p_right > p_left):
        return "<"
