#!/usr/bin/env python3

import os
import sys
from itertools import combinations, product

import identification as ide
import numpy as np
import pandas as pd
import pkg_resources

from etspredict.pipelines.ioutils import IOManager
from etspredict.prepare_data import prepare_connectomes as pc


def load_subject_list(dataset):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    stream = pkg_resources.resource_stream(__name__, path)
    return list(np.loadtxt(stream, dtype=str))


def identification_individual_bins(
    sessions,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
    full_fc,
):
    ranks_ = [f"Rank {i}" for i in ranks]
    ranks = ranks_

    results = {
        "threshold": [],
        "bin": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "ISR Bin-Full": [],
        "ISR Full-Bin": [],
    }

    binned_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions,
        preprocessing_level=preprocessing_level,
        function="all_slices",
        parcellation=parcellation,
        rank=ranks,
        criterion=criterion,
        threshold=threshold,
    )

    for rank in ranks:

        binned = binned_dict[rank]
        binned.columns = binned.columns.astype(str)

        bin_full = ide.identify(binned, full_fc, metric=correlation_method)
        full_bin = ide.identify(full_fc, binned, metric=correlation_method)

        idiff_mat = ide.get_idiff_matrix(
            binned, full_fc, metric=correlation_method
        )

        mask = np.ones(idiff_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        off_diag = idiff_mat.values[mask]

        mean_within = np.mean(np.diag(idiff_mat.values))
        mean_between = np.mean(off_diag)
        idiff = 100 * (mean_within - mean_between)

        results["threshold"].append(threshold)
        results["bin"].append(rank)
        results["mean within subjects"].append(mean_within)
        results["mean between subjects"].append(mean_between)
        results["idiff"].append(idiff)
        results["ISR Bin-Full"].append(bin_full)
        results["ISR Full-Bin"].append(full_bin)

    return pd.DataFrame(results)


def identification_combined_bins(
    sessions,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
    full_fc,
):

    rank_tuples = ["_".join([x, y]) for x, y in combinations(ranks, 2)]
    results = {
        "threshold": [],
        "bin1": [],
        "bin2": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "ISR Bin-Full": [],
        "ISR Full-Bin": [],
    }

    binned_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions,
        preprocessing_level=preprocessing_level,
        function="combined_bins",
        parcellation=parcellation,
        rank=rank_tuples,
        criterion=criterion,
        threshold=threshold,
    )

    for rank in rank_tuples:

        binned = binned_dict[rank]
        binned.columns = binned.columns.astype(str)

        bin_full = ide.identify(binned, full_fc, metric=correlation_method)
        full_bin = ide.identify(full_fc, binned, metric=correlation_method)

        idiff_mat = ide.get_idiff_matrix(
            binned, full_fc, metric=correlation_method
        )

        mask = np.ones(idiff_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        off_diag = idiff_mat.values[mask]

        mean_within = np.mean(np.diag(idiff_mat.values))
        mean_between = np.mean(off_diag)
        idiff = 100 * (mean_within - mean_between)

        bins = rank.split("_")
        results["threshold"].append(threshold)
        results["bin1"].append(bins[0])
        results["bin2"].append(bins[1])
        results["mean within subjects"].append(mean_within)
        results["mean between subjects"].append(mean_between)
        results["idiff"].append(idiff)
        results["ISR Bin-Full"].append(bin_full)
        results["ISR Full-Bin"].append(full_bin)

    return pd.DataFrame(results)


def identification_sequential(
    sessions,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
    full_fc,
):

    results = {
        "threshold": [],
        "amplitude magnitude": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "ISR Bin-Full": [],
        "ISR Full-Bin": [],
    }
    i = 0
    for thresh, magnitude in product(threshold, ranks):
        i += 1
        print(f"Running {i} OUT OF 24", end="\r")
        binned = pc.load_group_connectomes(
            dataset=dataset,
            subjects=subjects,
            sessions=sessions,
            preprocessing_level=preprocessing_level,
            function="high_low",
            parcellation=parcellation,
            rank=magnitude,
            criterion=criterion,
            threshold=thresh,
        )
        binned.columns = binned.columns.astype(str)

        idiff_mat = ide.get_idiff_matrix(
            binned, full_fc, metric=correlation_method
        )

        mask = np.ones(idiff_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        off_diag = idiff_mat.values[mask]

        mean_within = np.mean(np.diag(idiff_mat.values))
        mean_between = np.mean(off_diag)
        idiff = 100 * (mean_within - mean_between)

        binned_full = ide.identify(binned, full_fc, metric=correlation_method)
        full_binned = ide.identify(full_fc, binned, metric=correlation_method)

        results["threshold"].append(thresh)
        results["amplitude magnitude"].append(magnitude)
        results["mean within subjects"].append(mean_within)
        results["mean between subjects"].append(mean_between)
        results["idiff"].append(idiff)

        results["ISR Full-Bin"].append(binned_full)
        results["ISR Bin-Full"].append(full_binned)

        print(results)

    return pd.DataFrame(results)


def main(
    dataset,
    subjects,
    preprocessing_level="None",
    parcellation="schaefer200x17",
    criterion="rss",
    correlation_method="pearson",
    sampling_strat="individual_bins",
):

    thresholds_dict = {
        "hcp_aging": 12.5,
        "hcp": 5,
    }

    ranks_dict = {
        "hcp_aging": [f"{i}" for i in range(1, 9)],
        "hcp": [f"{i}" for i in range(1, 21)],
    }

    sessions_dict = {
        "hcp": (["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]),
        "hcp_aging": (["REST1_PA", "REST1_AP", "REST2_PA", "REST2_AP"]),
    }
    sessions = sessions_dict[dataset]

    if sampling_strat in ["individual_bins", "combined_bins"]:
        threshold = thresholds_dict[dataset]
        ranks = ranks_dict[dataset]
    elif sampling_strat in ["sequential"]:
        threshold = [1, 2, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
        ranks = ["high", "low"]
    else:
        raise NotImplementedError(
            f"{sampling_strat} not implemented as a sampling strategy!"
        )

    func_dict = {
        "individual_bins": identification_individual_bins,
        "combined_bins": identification_combined_bins,
        "sequential": identification_sequential,
    }

    full_fc = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        preprocessing_level=preprocessing_level,
        sessions=sessions,
        function="high_low",
        parcellation=parcellation,
        rank="high",
        criterion=criterion,
        threshold=100,
    )
    full_fc.columns = full_fc.columns.astype(str)

    return func_dict[sampling_strat](
        sessions,
        preprocessing_level,
        ranks,
        threshold,
        dataset,
        subjects,
        criterion,
        correlation_method,
        parcellation,
        full_fc,
    )


if __name__ == "__main__":

    dataset = sys.argv[1]
    preproc_level = sys.argv[2]
    parcellation = sys.argv[3]
    criterion = sys.argv[4]
    correlation_method = sys.argv[5]
    sampling_strat = sys.argv[6]

    subjects = load_subject_list(dataset)

    params = {
        "dataset": dataset,
        "subjects": subjects,
        "preprocessing_level": preproc_level,
        "parcellation": parcellation,
        "criterion": criterion,
        "correlation_method": correlation_method,
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
        path_elements=[
            f"idiff_{sampling_strat}_with_full_FC",
            dataset,
            preproc_level,
        ],
        file_parameters=outfile_parameters,
        save_index=False,
    )
    i_o.construct_output_directory()

    result = main(**params)
    i_o.save_dataframe(result)
