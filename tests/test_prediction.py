#!/usr/bin/env python3

import os
import tempfile
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold

from etspredict.prediction import model_grids, pipe, scoring_utils

np.random.seed(1234567)


#############################################################################
# test all functions in scoring_utils.py
#############################################################################


def test_avg_cv_scores():
    cv_scores = pd.DataFrame(
        {
            "Fold1": [0.9, 0.5, 0.6, 0.7],
            "Fold2": [0.4, 0.3, 0.9, 0.4],
            "Fold3": [0.8, 0.5, 0.8, 0.8],
            "Fold4": [0.8, 0.5, 0.4, 0.9],
            "Fold5": [0.9, 0.5, 0.7, 0.3],
        }
    )
    avg_score, std = scoring_utils.avg_cv_scores(cv_scores)

    assert round(avg_score, 2) == 0.63
    assert round(std, 2) == 0.13

    cv_scores = pd.DataFrame(
        {
            "Fold1": [0.9],
            "Fold2": [0.4],
            "Fold3": [0.8],
            "Fold4": [0.8],
            "Fold5": [0.9],
        }
    )
    avg_score, std = scoring_utils.avg_cv_scores(cv_scores)

    assert round(avg_score, 2) == 0.76
    assert round(std, 2) == 0.19


#############################################################################
# test all functions in model_grids.py
#############################################################################


def test_get_model_grid():
    problems = ["classification", "regression"]
    models = ["dummy", "lin"]

    for problem, model in product(problems, models):

        ret_model, ret_params = model_grids.get_model_grid(model, problem)
        if problem in ["classification"]:
            assert ret_model in [
                "logit",
                "dummy",
            ]
        elif problem in ["regression"]:
            assert ret_model in ["linreg", "dummy"]

        assert isinstance(ret_params, dict)

    ret_model, ret_params = model_grids.get_model_grid(
        "kernelridge_scikit", "regression"
    )
    assert isinstance(ret_model, KernelRidge)
    assert isinstance(ret_params, dict)
    assert "kernelridge__alpha" in ret_params.keys()
    assert isinstance(ret_params["kernelridge__alpha"], list)

    ret_model, ret_params = model_grids.get_model_grid(
        "ridgeclassifier", "classification"
    )
    assert isinstance(ret_model, RidgeClassifier)
    assert isinstance(ret_params, dict)
    assert "ridgeclassifier__alpha" in ret_params.keys()
    assert isinstance(ret_params["ridgeclassifier__alpha"], list)


#############################################################################
# test all functions in pipe.py
#############################################################################


def test_pipeline():

    with tempfile.TemporaryDirectory() as tmpdirname:

        problem_type = "regression"

        # simulate features
        feature_df = pd.DataFrame(
            np.random.randn(100, 4), columns=list("ABCD")
        )
        features = list("ABCD")

        # simulate a target
        feature_df["test_target"] = np.random.rand(100)

        # simulate confounds
        for conf in ["Confound1", "Confound2"]:
            feature_df[conf] = np.random.rand(100)

        # simulate a grouping variable
        feature_df["TESTGROUP"] = np.random.randint(0, 5, size=(100))

        assert set(feature_df["TESTGROUP"]) == set([0, 1, 2, 3, 4])

        print(feature_df)

        ret_model, ret_params = model_grids.get_model_grid("lin", problem_type)
        pipeline = pipe.Pipeline(
            model=ret_model,
            model_params=ret_params,
            pipeline_name="test_pipeline",
            output_directory=tmpdirname,
            features=features,
            target="test_target",
            confounds=["Confound1", "Confound2"],
            prediction_data=feature_df,
            problem_type=problem_type,
            preprocess_X=None,
            preprocess_y="remove_confound",
            return_estimator="final",
            scoring="corr",
            seed=1234567,
            cv=GroupKFold(n_splits=4),
            groups="TESTGROUP",
        )

        pipeline.run_julearn()
        pipeline.save_model()
        pipeline.save_scores()
        pipeline.save_timelog()
        pipeline.save_deconf_info()

        # test if all output is being saved
        file_extensions = [".txt", ".csv", ".sav"]
        for f_type in file_extensions:
            f_name = os.path.join(tmpdirname, f"test_pipeline{f_type}")
            assert os.path.isfile(f_name)

        deconf_file = os.path.join(
            tmpdirname, f"test_pipeline_deconf_info.csv"
        )
        assert os.path.isfile(deconf_file)


def test_scorers():

    var_a = np.random.rand(100)
    var_b = np.random.rand(100)
    assert pipe.corr(var_a, var_b) == pearsonr(var_a, var_b)[0]
    assert pipe.spear_corr(var_a, var_b) == spearmanr(var_a, var_b)[0]
