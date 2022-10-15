#!/usr/bin/env python3

import os

import pandas as pd


def configure_output_name(target, sampling_strat, sc_free):
    if target in ["sex", "Gender"]:
        name = f"sex_predictions_{sampling_strat}"
    elif target in ["age", "Age_in_Yrs"]:
        name = f"age_predictions_{sampling_strat}"
    else:
        name = f"predictions_{sampling_strat}"

    if sc_free:
        name = f"{name}_sc_free"

    return name


class IOManager:
    def __init__(
        self, root_dir, path_elements, file_parameters, save_index=True
    ):
        self.output_directory = root_dir
        self.path_elements = path_elements
        self.file_parameters = file_parameters
        self.save_index = save_index

    def construct_output_directory(self):
        """Constructs an output directory according to a list of folders.

        Parameters
        -----------
        path_elements : list
            strings, each of which define a folder. Order defines the folder
            hierarchy

        """

        for element in self.path_elements:
            assert (
                self.output_directory is not None
            ), "Specified root directory is NoneType"
            self.output_directory = os.path.join(
                self.output_directory, element
            )

        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def save_dataframe(self, dataframe):
        """Takes a dataframe and saves it as a file named after parameters
        specified in a dictionary (similar to bids format)

        """

        i = 0
        fname = ""
        for key, value in self.file_parameters.items():
            if i > 0:
                fname = f"{fname}_{key}-{value}"
            else:
                fname = f"{key}-{value}"
            i += 1
        fname = f"{fname}.tsv"

        self.outfile = os.path.join(self.output_directory, fname)
        assert isinstance(
            dataframe, pd.DataFrame
        ), "Provide a valid dataframe to save_dataframe!"
        dataframe.to_csv(self.outfile, sep="\t", index=self.save_index)
        print(f"Output has been saved at {self.outfile}!")
