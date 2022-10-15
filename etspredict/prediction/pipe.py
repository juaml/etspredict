#!/usr/bin/env python3


import os
from datetime import datetime

import joblib
import pandas as pd
from julearn import run_cross_validation
from julearn.scoring import register_scorer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import make_scorer

os.environ["OPENBLAS_NUM_THREADS"] = "1"


# -------------------------------------------------------------------------- #
# add pearson correlation as a valid sklearn scorer


def corr(y_true, y_pred):

    return pearsonr(y_true, y_pred)[0]


def spear_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


sklearn_scorer = make_scorer(corr)
spearman_scorer = make_scorer(spear_corr)

register_scorer("spear_corr", spearman_scorer)
register_scorer("corr", sklearn_scorer)


# -------------------------------------------------------------------------- #


class Pipeline:
    def __init__(
        self,
        model,
        model_params=None,
        pipeline_name=None,
        output_directory=None,
        features=None,
        target=None,
        confounds=None,
        prediction_data=None,
        problem_type=None,
        preprocess_X=None,
        preprocess_y=None,
        return_estimator=None,
        scoring=None,
        seed=1234567,
        cv=None,
        groups=None,
    ):
        """Initialise a CV Pipeline

        Parameters
        -----------
        model : str or Sklearn-estimator
            If model is a string it should be one of the available models
            in julearn.
        model_params : dict
            Dictionary with model parameters for a grid search
        pipeline_name : str
            Unique identifier for a given pipeline to be used when saving
            results
        output_directory : str or path
            Path to an existing directory in which pipeline results should be
            saved
        features : str or list of str
            Name of columns of features in prediction_data to be passed on to
            Julearn
        target : str
            Name of column which contains target to predict
        confounds : str or list of str
            Names of columns which contain confounding variables in
            prediction_data
        prediction_data : pd.DataFrame
            DataFrame containing features, (optionally) confounds, and target
        problem_type : str
            Julearn-compatible problem types, i.e. "regression" or
            "classification"
        preprocess_X : str
            How Julearn should preprocess the features
        preprocess_y : str
            How Julearn should preprocess the target
        return_estimator : None or str
            None, "final" or "all"
        scoring : str or julearn-registered sklearn scorer
            Any Sklearn compatible scorer
        seed : int
            which seed to use for randomisation
        zscore_rows : Boolean
            Whether to zscore the features row-wise (i.e. for each sample)

        """
        self.model = model
        self.model_params = model_params
        self.output_directory = output_directory
        self.pipeline_name = pipeline_name

        # ------------------------------------------------------------------ #
        # Initialise Julearn

        self.features = features
        self.target = target
        self.confounds = confounds
        self.prediction_data = prediction_data
        self.problem_type = problem_type
        self.preprocess_X = preprocess_X
        self.preprocess_y = preprocess_y
        self.return_estimator = return_estimator
        self.scoring = scoring
        self.model_params = model_params
        self.seed = seed
        self.cv = cv
        self.groups = groups

    def run_julearn(self):
        """Run Julearn for nested CV to get scores and a model"""

        totaltime = datetime.timestamp(datetime.now())

        assert isinstance(
            self.pipeline_name, str
        ), "Please provide a valid name for your pipeline!"
        try:
            assert os.path.isdir(
                self.output_directory
            ), "Please provide the name of an existing output directory!"
        except TypeError:
            raise AssertionError(
                "Please provide the name of an existing output directory!"
            )

        n_before = self.prediction_data.shape[0]
        self.prediction_data.dropna(axis=0, inplace=True)
        n_after = self.prediction_data.shape[0]

        dropped = n_before - n_after
        print("------------------------------------------------------------\n")
        print(f"{dropped} subjects were removed due to NaN values \n")
        print("------------------------------------------------------------\n")
        print(f"problem_type        ==      {self.problem_type}")
        print(f"model               ==      {self.model}")
        print(f"model_params        ==      {self.model_params}")
        print(f"preprocess_X        ==      {self.preprocess_X}")
        print(f"preprocess_y        ==      {self.preprocess_y}")
        print(f"return_estimator    ==      {self.return_estimator}")
        print(f"scoring             ==      {self.scoring}")
        print(f"seed                ==      {self.seed}")
        print(f"groupbed by:        ==      {self.groups}")
        print(f"cv                  ==      {self.cv}")
        print("------------------------------------------------------------\n")
        print("Starting nested CV using Julearn\n")
        print("------------------------------------------------------------\n")

        self.scores, self.model = run_cross_validation(
            X=self.features,
            y=self.target,
            confounds=self.confounds,
            data=self.prediction_data,
            problem_type=self.problem_type,
            model=self.model,
            model_params=self.model_params,
            preprocess_X=self.preprocess_X,
            preprocess_y=self.preprocess_y,
            return_estimator=self.return_estimator,
            scoring=self.scoring,
            seed=self.seed,
            cv=self.cv,
            groups=self.groups,
        )

        print("------------------------------------------------------------\n")
        print("Julearn is Done!\n")
        print("------------------------------------------------------------\n")

        stoptime = datetime.timestamp(datetime.now())
        self.time = stoptime - totaltime

    def save_model(self):
        """Save the ExtendedDataFramePipeline returned by Julearn as a .sav
        file using joblib.

        """

        file_name = os.path.join(
            self.output_directory, f"{self.pipeline_name}.sav"
        )
        print("Saving model at:\n")
        print("------------------------------------------------------------\n")
        print(f"{file_name}\n")
        print("------------------------------------------------------------\n")
        joblib.dump(self.model, file_name)

    def save_scores(self):
        """Save scores from CV as a .csv file."""

        file_name = os.path.join(
            self.output_directory, f"{self.pipeline_name}.csv"
        )
        print("Saving scores at:\n")
        print("------------------------------------------------------------\n")
        print(f"{file_name}\n")
        print("------------------------------------------------------------\n")
        self.scores.to_csv(file_name, index=False)

    def save_timelog(self):
        """Save the time pipeline took in seconds as a .txt file."""
        file_name = os.path.join(
            self.output_directory, f"{self.pipeline_name}.txt"
        )
        print("Saving timelog at:\n")
        print("------------------------------------------------------------\n")
        print(f"{file_name}\n")
        print("------------------------------------------------------------\n")

        with open(file_name, "w") as f:
            f.write(f"{self.time}")

    def save_deconf_info(self):
        """Save confounds, original target, deconfounded target, and predicted
        target for all the data in a single .csv file.

        """
        file_name = os.path.join(
            self.output_directory, f"{self.pipeline_name}_deconf_info.csv"
        )
        data_to_predict = pd.concat(
            [
                self.prediction_data[self.features],
                self.prediction_data[self.confounds],
            ],
            axis=1,
        )
        y_pred = self.model.predict(data_to_predict)

        _, y_deconf = self.model.preprocess(
            data_to_predict,
            self.prediction_data[self.target],
            until="remove_confound",
        )

        output = self.prediction_data[self.confounds]
        output[f"{self.target}_original"] = self.prediction_data[self.target]
        output[f"{self.target}_predicted"] = y_pred
        output[f"{self.target}_deconfounded"] = y_deconf
        output.to_csv(file_name, index=False)
