#!/usr/bin/env python3
# -------------------------------------------------------------------------- #


import os

import pandas as pd
import pkg_resources


def load_info_targets(category=None, column=None):
    """A function to retrieve and display some information about 58
    pre-selected possible targets from the HCP behavioural data. This selection
    is equivalent to the targets used in:

    https://www.biorxiv.org/content/10.1101/2021.01.16.426943v2
    """

    data_diary_path = os.path.join("pheno", "hcp_data_diary_targets.csv")
    stream = pkg_resources.resource_stream(__name__, data_diary_path)

    data_diary = pd.read_csv(stream)
    categories = data_diary["category"].value_counts().index

    if category is not None:
        if isinstance(category, str):
            category = [category]

        data_list = []
        for cat in category:
            assert cat in categories, (
                "Your specified category is not available, available"
                f" categories include {categories}"
            )
            cat_mask = data_diary["category"] == cat
            data_list.append(data_diary.loc[cat_mask])

        data_diary = pd.concat(data_list)

    return data_diary if column is None else data_diary[column]


def get_restricted_data(columns=None, subjects=None):

    restricted_path = os.path.join("pheno", "restricted.csv")
    stream = pkg_resources.resource_stream(__name__, restricted_path)

    restricted_data = pd.read_csv(stream, dtype={"Subject": str})
    restricted_data.set_index("Subject", inplace=True)

    if columns is not None:
        restricted_data = restricted_data[columns]

    if subjects is not None:
        if isinstance(subjects, list):
            subjects = list(map(str, subjects))
        restricted_data = restricted_data.loc[subjects]

    return restricted_data


def get_unrestricted_data(columns=None, subjects=None):

    unrestricted_path = os.path.join("pheno", "unrestricted.csv")
    stream = pkg_resources.resource_stream(__name__, unrestricted_path)
    unrestricted_data = pd.read_csv(stream, dtype={"Subject": str})
    unrestricted_data.set_index("Subject", inplace=True)

    if columns is not None:
        unrestricted_data = unrestricted_data[columns]

    if subjects is not None:
        if isinstance(subjects, list):
            subjects = list(map(str, subjects))
        unrestricted_data = unrestricted_data.loc[subjects]

    return unrestricted_data


def load_behavioural_data(datafields, subjects=None):

    unrestricted_data = get_unrestricted_data(subjects=subjects)
    restricted_data = get_restricted_data(subjects=subjects)

    if isinstance(datafields, str):
        datafields = [datafields]

    assert isinstance(datafields, list), "Provide confounds as string or list!"

    restricted_fields = []
    unrestricted_fields = []

    for field in datafields:
        if field in unrestricted_data.columns:
            unrestricted_fields.append(field)
        elif field in restricted_data.columns:
            restricted_fields.append(field)

    combined_data = pd.DataFrame()
    if restricted_fields:
        combined_data[restricted_fields] = restricted_data[restricted_fields]
    if unrestricted_fields:
        combined_data[unrestricted_fields] = unrestricted_data[
            unrestricted_fields
        ]

    if "Acquisition" in combined_data.columns:
        datapoints = []
        for i in combined_data["Acquisition"]:
            datapoints.append(int(i[1:]))

        combined_data["Acquisition"] = datapoints

    return combined_data


def load_fd_data(subjects=None, sessions=None):

    fd_data_path = os.path.join("pheno", "hcp_ya_fd.csv")

    stream = pkg_resources.resource_stream(__name__, fd_data_path)
    fd_data = pd.read_csv(stream, dtype={"subjects": str})
    fd_data.set_index("subjects", inplace=True)

    if subjects is not None:
        fd_data = fd_data.loc[subjects]
    if sessions is not None:
        fd_data = fd_data[sessions]

    return fd_data


def load_vol_TIV_data(subjects=None):

    vol_TIV_data_path = os.path.join("pheno", "vol_TIV_HCP_397subs.csv")
    stream = pkg_resources.resource_stream(__name__, vol_TIV_data_path)
    vol_TIV_data = pd.read_csv(stream, dtype={"subjects": str})
    vol_TIV_data.set_index("subjects", inplace=True)

    return vol_TIV_data
