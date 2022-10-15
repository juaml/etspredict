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
    sessions1,
    sessions2,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
):
    save_args = locals()
    save_args.pop("ranks", None)
    save_args.pop("sessions1", None)
    save_args.pop("sessions2", None)
    save_args.pop("subjects", None)

    name_idiff_matrix = [f"{x}-{y}" for x, y in save_args.items()]
    name_idiff_matrix = "_".join(name_idiff_matrix)
    name_idiff_matrix = f"individual_bins_{name_idiff_matrix}"

    ranks_ = [f"Rank {i}" for i in ranks]
    ranks = ranks_

    results = {
        "threshold": [],
        "bin": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "ISR REST1-REST2": [],
        "ISR REST2-REST1": [],
    }

    rest1_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions1,
        preprocessing_level=preprocessing_level,
        function="all_slices",
        parcellation=parcellation,
        rank=ranks,
        criterion=criterion,
        threshold=threshold,
    )

    rest2_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions2,
        preprocessing_level=preprocessing_level,
        function="all_slices",
        parcellation=parcellation,
        rank=ranks,
        criterion=criterion,
        threshold=threshold,
    )

    for rank in ranks:

        rest1 = rest1_dict[rank]
        rest2 = rest2_dict[rank]

        r1r2 = ide.identify(rest1, rest2, metric=correlation_method)
        r2r1 = ide.identify(rest2, rest1, metric=correlation_method)

        idiff_mat = ide.get_idiff_matrix(
            rest1, rest2, metric=correlation_method
        )

        name_idiff_matrix_rank = f"{name_idiff_matrix}_{rank}"
        idiff_mat.to_csv(f"../results/idiff_matrices/{name_idiff_matrix_rank}")

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
        results["ISR REST1-REST2"].append(r1r2)
        results["ISR REST2-REST1"].append(r2r1)

    return pd.DataFrame(results)


def identification_combined_bins(
    sessions1,
    sessions2,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
):

    rank_tuples = ["_".join([x, y]) for x, y in combinations(ranks, 2)]
    results = {
        "threshold": [],
        "bin1": [],
        "bin2": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "ISR REST1-REST2": [],
        "ISR REST2-REST1": [],
    }

    rest1_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions1,
        preprocessing_level=preprocessing_level,
        function="combined_bins",
        parcellation=parcellation,
        rank=rank_tuples,
        criterion=criterion,
        threshold=threshold,
    )

    rest2_dict = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        sessions=sessions2,
        preprocessing_level=preprocessing_level,
        function="combined_bins",
        parcellation=parcellation,
        rank=rank_tuples,
        criterion=criterion,
        threshold=threshold,
    )

    for rank in rank_tuples:

        rest1 = rest1_dict[rank]
        rest2 = rest2_dict[rank]

        r1r2 = ide.identify(rest1, rest2, metric=correlation_method)
        r2r1 = ide.identify(rest2, rest1, metric=correlation_method)

        idiff_mat = ide.get_idiff_matrix(
            rest1, rest2, metric=correlation_method
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
        results["ISR REST1-REST2"].append(r1r2)
        results["ISR REST2-REST1"].append(r2r1)

    return pd.DataFrame(results)


def identification_sequential(
    sessions1,
    sessions2,
    preprocessing_level,
    ranks,
    threshold,
    dataset,
    subjects,
    criterion,
    correlation_method,
    parcellation,
):

    results = {
        "threshold": [],
        "amplitude magnitude": [],
        "mean within subjects": [],
        "mean between subjects": [],
        "idiff": [],
        "identification_accuracy_r1r2": [],
        "identification_accuracy_r2r1": [],
    }
    i = 0
    for thresh, magnitude in product(threshold, ranks):
        i += 1
        print(f"Running {i} OUT OF 24", end="\r")
        rest1 = pc.load_group_connectomes(
            dataset=dataset,
            subjects=subjects,
            sessions=sessions1,
            preprocessing_level=preprocessing_level,
            function="high_low",
            parcellation=parcellation,
            rank=magnitude,
            criterion=criterion,
            threshold=thresh,
        )
        print(rest1)
        rest2 = pc.load_group_connectomes(
            dataset=dataset,
            subjects=subjects,
            sessions=sessions2,
            preprocessing_level=preprocessing_level,
            function="high_low",
            parcellation=parcellation,
            rank=magnitude,
            criterion=criterion,
            threshold=thresh,
        )
        print(rest2)

        idiff_mat = ide.get_idiff_matrix(
            rest1, rest2, metric=correlation_method
        )

        mask = np.ones(idiff_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        off_diag = idiff_mat.values[mask]

        mean_within = np.mean(np.diag(idiff_mat.values))
        mean_between = np.mean(off_diag)
        idiff = 100 * (mean_within - mean_between)

        r1r2 = ide.identify(rest1, rest2, metric=correlation_method)
        r2r1 = ide.identify(rest2, rest1, metric=correlation_method)

        results["threshold"].append(thresh)
        results["amplitude magnitude"].append(magnitude)
        results["mean within subjects"].append(mean_within)
        results["mean between subjects"].append(mean_between)
        results["idiff"].append(idiff)

        results["identification_accuracy_r1r2"].append(r1r2)
        results["identification_accuracy_r2r1"].append(r2r1)

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
        "hcp": (["REST1_LR", "REST1_RL"], ["REST2_LR", "REST2_RL"]),
        "hcp_aging": (["REST1_PA", "REST1_AP"], ["REST2_PA", "REST2_AP"]),
    }
    sessions1, sessions2 = sessions_dict[dataset]

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

    return func_dict[sampling_strat](
        sessions1,
        sessions2,
        preprocessing_level,
        ranks,
        threshold,
        dataset,
        subjects,
        criterion,
        correlation_method,
        parcellation,
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
        path_elements=[f"idiff_{sampling_strat}", dataset, preproc_level],
        file_parameters=outfile_parameters,
        save_index=False,
    )
    i_o.construct_output_directory()

    result = main(**params)
    i_o.save_dataframe(result)
