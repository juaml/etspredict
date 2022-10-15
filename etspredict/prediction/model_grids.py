#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:29:12 2021
@author: leonard
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier

lambdas = [
    0,
    0.00001,
    0.0001,
    0.001,
    0.004,
    0.007,
    0.01,
    0.04,
    0.07,
    0.1,
    0.4,
    0.7,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    100,
    150,
    200,
    300,
    500,
    700,
    1000,
    10000,
    100000,
    1000000,
]

c_vals = np.geomspace(1e-2, 1e2, 50)

# Dict returning for each model the julearn name + params
model_dict_classification = {
    "lin": ["logit", {"logit__C": c_vals, "logit__max_iter": [100_000]}],
    "rbf_svm": [
        "svm",
        {
            "svm__kernel": ["rbf"],
            "svm__probability": [True],
            "svm__C": c_vals,
        },
    ],
    "linear_svm": [
        "svm",
        {
            "svm__kernel": ["linear"],
            "svm__probability": [True],
            "svm__C": c_vals,
        },
    ],
    "dummy": ["dummy", {"dummy__strategy": ["prior"]}],
    "ridgeclassifier": [
        RidgeClassifier(),
        {"ridgeclassifier__alpha": lambdas},
    ],
}

model_dict_regression = {
    "lin": ["linreg", {}],
    "dummy": ["dummy", {}],
    "kernelridge_scikit": [KernelRidge(), {"kernelridge__alpha": lambdas}],
}


def get_model_grid(model_name, problem_type):
    """
    Parameters
    ----------
    model_name : string
        For which model to retrieve gridsearch parameters.
    problem_type : String
        "regression" or "classification".
    Raises
    ------
    ValueError
        DESCRIPTION.
    Returns
    -------
    model : String
        model to pass to run_cross_validation().
    param_dict : dict
        params to pass over to grid search.
    """
    if problem_type == "regression":
        model, param_dict = model_dict_regression[model_name]
    else:
        model, param_dict = model_dict_classification[model_name]

    print("----------------------------------------------------------------\n")
    print(f"The selected model is: {model}\n")
    print("Model Parameters are as follows:\n")
    for key, value in param_dict.items():
        print(f"{key} ==== {value}\n")
    print("----------------------------------------------------------------\n")
    return model, param_dict
