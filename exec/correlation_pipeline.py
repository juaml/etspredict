#!/usr/bin/env python3

import os
import sys
from itertools import combinations, product

import numpy as np
import pandas as pd
import pkg_resources

from etspredict.data import load_struct_connectomes
from etspredict.pipelines.ioutils import IOManager
from etspredict.prepare_data import prepare_connectomes as pc


def load_subject_list(dataset):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    stream = pkg_resources.resource_stream(__name__, path)
    return list(np.loadtxt(stream, dtype=str))


def correlation_individual_bins(
    full_fc,
    df_sc,
    subs_to_exclude,
    ranks,
    corrmethod,
    threshold,
    preprocessing_level,
    subjects,
):

    fc_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
        preprocessing_level=preprocessing_level,
        function="all_slices",
        parcellation=parcellation,
        rank=ranks,
        criterion=criterion,
        threshold=threshold,
    )

    results = pd.DataFrame()
    # correlation full fc and sc
    results["full"] = df_sc.corrwith(full_fc, axis=1, method=corrmethod)

    for rank in ranks:
        bin_fc = fc_dict[rank]
        bin_fc = bin_fc.drop(subs_to_exclude)
        bin_fc.columns = bin_fc.columns.astype(str)
        results[rank] = df_sc.corrwith(bin_fc, axis=1, method=corrmethod)

    return results


def correlation_combined_bins(
    full_fc,
    df_sc,
    subs_to_exclude,
    ranks,
    corrmethod,
    threshold,
    preprocessing_level,
    subjects,
):
    rank_tuples = ["_".join([x, y]) for x, y in combinations(ranks, 2)]

    fc_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
        preprocessing_level=preprocessing_level,
        function="combined_bins",
        parcellation=parcellation,
        rank=rank_tuples,
        criterion=criterion,
        threshold=threshold,
    )

    results = pd.DataFrame()
    # correlation full fc and sc
    results["full"] = df_sc.corrwith(full_fc, axis=1, method=corrmethod)

    for rank in rank_tuples:
        bin_fc = fc_dict[rank]
        bin_fc = bin_fc.drop(subs_to_exclude)
        bin_fc.columns = bin_fc.columns.astype(str)
        results[rank] = df_sc.corrwith(bin_fc, axis=1, method=corrmethod)

    return results


def correlation_sequential(
    full_fc,
    df_sc,
    subs_to_exclude,
    ranks,
    corrmethod,
    threshold,
    preprocessing_level,
    subjects,
):

    results = pd.DataFrame()
    # correlation full fc and sc
    results["full"] = df_sc.corrwith(full_fc, axis=1, method=corrmethod)

    i = 0
    fc_dict = {}
    for thresh, magnitude in product(threshold, ranks):
        i += 1
        print(f"Running {i} OUT OF 24", end="\r")
        fc = pc.load_group_connectomes(
            dataset=dataset,
            subjects=subjects,
            sessions=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
            preprocessing_level=preprocessing_level,
            function="high_low",
            parcellation=parcellation,
            rank=magnitude,
            criterion=criterion,
            threshold=thresh,
        )
        fc_dict[f"{thresh}_{magnitude}"] = fc

    for rank, bin_fc in fc_dict.items():
        bin_fc = bin_fc.drop(subs_to_exclude)
        bin_fc.columns = bin_fc.columns.astype(str)
        results[rank] = df_sc.corrwith(bin_fc, axis=1, method=corrmethod)

    return results


def main(
    dataset,
    subjects,
    preprocessing_level="None",
    parcellation="schaefer200x17",
    criterion="rss",
    corrmethod="pearson",
    sampling_strat="individual_bins",
):

    if dataset not in ["hcp"]:
        raise NotImplementedError("Only dataset is hcp")

    threshold_dict = {
        "individual_bins": 5,
        "combined_bins": 5,
        "sequential": [1, 2, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100],
    }
    threshold = threshold_dict[sampling_strat]

    ranks_dict = {
        "individual_bins": [f"Rank {i}" for i in range(1, 21)],
        "combined_bins": [f"{i}" for i in range(1, 21)],
        "sequential": ["high", "low"],
    }
    ranks = ranks_dict[sampling_strat]

    funcs_dict = {
        "individual_bins": correlation_individual_bins,
        "combined_bins": correlation_combined_bins,
        "sequential": correlation_sequential,
    }

    print("Loading full connectome")
    full_fc = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        preprocessing_level=preprocessing_level,
        sessions=["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
        function="high_low",
        parcellation=parcellation,
        rank="high",
        criterion=criterion,
        threshold=100,
    )

    df_sc = load_struct_connectomes(parcellation)
    df_sc.index = df_sc.index.astype(str)
    df_sc.dropna(inplace=True)
    subs_to_exclude = [x for x in full_fc.index if x not in df_sc.index]
    full_fc = full_fc.drop(index=subs_to_exclude)
    df_sc = df_sc.loc[full_fc.index]

    return funcs_dict[sampling_strat](
        full_fc,
        df_sc,
        subs_to_exclude,
        ranks,
        corrmethod,
        threshold,
        preprocessing_level,
        subjects,
    )


if __name__ == "__main__":

    dataset = sys.argv[1]
    preproc_level = sys.argv[2]
    parcellation = sys.argv[3]
    criterion = "rss"  # sys.argv[4]
    corrmethod = sys.argv[4]
    sampling_strat = sys.argv[5]

    subjects = load_subject_list(dataset)

    params = {
        "dataset": dataset,
        "subjects": subjects,
        "preprocessing_level": preproc_level,
        "parcellation": parcellation,
        "criterion": criterion,
        "corrmethod": corrmethod,
        "sampling_strat": sampling_strat,
    }
    outfile_parameters = {}
    for key, value in params.items():
        if isinstance(value, str) or isinstance(value, int):
            if key not in ["path_to_root"]:
                outfile_parameters[key] = value

    # construct appropriate outdir and file
    i_o = IOManager(
        root_dir=os.path.join("..", "results"),
        path_elements=[f"SC_FC_corr_{sampling_strat}", dataset, preproc_level],
        file_parameters=outfile_parameters,
        save_index=False,
    )
    i_o.construct_output_directory()

    result = main(**params)
    i_o.save_dataframe(result)
