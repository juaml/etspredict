import os
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources


class HCPAgingTimeseries:
    def __init__(self):
        self.timeseries = None

    def available_subjects(self, unrelated=True):
        parent = Path(__file__).parent.resolve()
        path = os.path.join("timeseries")
        file_list = os.listdir(parent / path)

        file_list = os.listdir(Path(__file__).parent.resolve() / path)
        subjects_all = [
            Path(x).stem
            for x in file_list
            if x not in ["HCA_LS_2.0_subject_completeness.csv"]
        ]

        if unrelated:
            path_compl = os.path.join(
                "timeseries", "HCA_LS_2.0_subject_completeness.csv"
            )
            stream = pkg_resources.resource_stream(__name__, path_compl)
            completeness_df = pd.read_csv(stream)
            related_subset_mask = (
                completeness_df["unrelated_subset"] == "FALSE"
            )
            related_subset = completeness_df.loc[related_subset_mask]
            related_subjects = list(related_subset["src_subject_id"])

            subjects = [x for x in subjects_all if x not in related_subjects]
        else:
            subjects = subjects_all

        return subjects

    def load(
        self,
        subject,
        session="REST1_AP",
        parcellation="schaefer200x17",
        preprocessing="GS",
    ):
        path = os.path.join(
            "timeseries",
            subject,
            preprocessing,
            parcellation,
            f"{subject}_{session}.npy",
        )
        stream = pkg_resources.resource_stream(__name__, path)
        self.timeseries = np.load(stream)


if __name__ == "__main__":
    a = HCPAgingTimeseries()
    a.available_subjects()
