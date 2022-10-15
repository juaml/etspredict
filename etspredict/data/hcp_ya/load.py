import os
from pathlib import Path

import numpy as np
import pkg_resources


class HCPTimeseries:
    def __init__(self):
        self.timeseries = None

    def available_subjects(self):
        path = os.path.join("timeseries")
        file_list = os.listdir(Path(__file__).parent.resolve() / path)

        file_list = os.listdir(Path(__file__).parent.resolve() / path)
        return [Path(x).stem for x in file_list]

    def load(
        self,
        subject,
        session="REST1_LR",
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
