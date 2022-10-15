#!/usr/bin/env python3

import time

import bct
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from nilearn.connectome import vec_to_sym_matrix
from scipy.stats import zscore


def get_edge_timeseries(timeseries):
    """This function will take a timeseries of brain areas, and calculate
    timeseries for each edge according to the method outlined by Betzel et al.
    (2020) -- (https://www.pnas.org/content/117/45/28393#sec-21). For their
    code see https://github.com/brain-networks/edge-ts/blob/master/main.m

    Parameters
    ----------
    timeseries : np.array
        number of rows should correspond to number of time frames
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)

    Returns
    -------
    ets, timeseries : tuple (np.array, np.array)
        [0] Edge-timeseries with rows corresponding to time points and columns
            corresponding to unique edges
        [1] either original timeseries, or deconfounded if so specified
    """

    timeseries = zscore(timeseries)

    _, n_roi = timeseries.shape

    # indices of unique edges (lower triangle)
    u, v = np.tril_indices(n_roi, k=-1)

    return timeseries[:, u] * timeseries[:, v]


def global_measures(timeseries):
    """Extracts time-varying global graph measures from the timepoint x nodes
    timeseries leveraging its edge-timeseries.

    Parameters
    ----------
    timeseries : np.array
        number of rows should correspond to number of time frames
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)

    returns
    -------
    measure_timeseries : pd.DataFrame
        Dataframe containing each graph measure extracted (columns) at each
        timeoint (rows)
    """

    ets = get_edge_timeseries(timeseries)
    n_timepoints, n_roi = timeseries.shape

    measure_functions = {
        "transitivity": bct.transitivity_wu,
        "weighted_clustering_coefficient": bct.clustering_coef_wu,
    }

    measure_timeseries = {}
    charpath_measures = ["charpath", "efficiency"]
    for measure in charpath_measures:
        measure_timeseries[measure] = []

    function_names = [x for x in measure_functions.keys()]
    for measure in function_names:
        measure_timeseries[measure] = []

    measure_names = [x for x in measure_timeseries.keys()]
    for measure in measure_names:
        measure_timeseries[f"{measure}_abs"] = []

    # get rss then change ets to pandas dataframe
    measure_timeseries["rss"] = np.sum(ets**2, 1) ** 0.5
    ets = pd.DataFrame(ets)

    for label, current_connectome in ets.iterrows():
        start = time.time()

        # nxn connectome:
        nxnconn = vec_to_sym_matrix(
            np.array(current_connectome), np.ones(n_roi)
        )

        # first get straightforward functions:
        for measure, function in measure_functions.items():
            measure_timeseries[measure].append(np.mean(function(nxnconn)))

        # now for absolute values
        for measure, function in measure_functions.items():
            measure_timeseries[f"{measure}_abs"].append(
                np.mean(function(np.abs(nxnconn)))
            )

        # then charpath functions
        D, _ = bct.distance_wei(nxnconn)
        D_abs, _ = bct.distance_wei(np.abs(nxnconn))

        cpl, efficiency, ecc, radius, diam = bct.charpath(D)
        cpl_abs, efficiency_abs, ecc_abs, radius_abs, diam_abs = bct.charpath(
            D_abs
        )

        measure_timeseries["charpath"].append(cpl)
        measure_timeseries["charpath_abs"].append(cpl_abs)
        measure_timeseries["efficiency"].append(efficiency)
        measure_timeseries["efficiency_abs"].append(efficiency_abs)

        stop = time.time()
        duration = stop - start
        print(
            f"Round {label + 1}/{n_timepoints} took {duration} seconds!",
            end="\r",
        )

    return pd.DataFrame(measure_timeseries)


def principal_gradient(
    timeseries, kernel="gaussian", approach="pca", random_state=0, sparsity=0
):
    """Extracts time-varying principal gradient from the timepoint x nodes
    timeseries leveraging its edge-timeseries.

    Parameters
    ----------
    timeseries : np.array
        number of rows should correspond to number of time frames
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    kernel : str
         Kernel function to build the affinity matrix.
         Possible options: {‘pearson’, ‘spearman’, ‘cosine’,
         ‘normalized_angle’, ‘gaussian’}. If callable, must receive a
         2D array and return a 2D square array. If None, use input matrix.
         Default is None.
    approach : str
        Embedding approach. Options are "pca", "dm", or "le".

    returns
    -------
    gradient_timeseries : pd.DataFrame
        Dataframe containing for each ROI (columns) the princpal gradient
        position (rows)

    """

    ets = get_edge_timeseries(timeseries)
    n_timepoints, n_roi = timeseries.shape

    # get an average gradient for alignment
    avg_conn = vec_to_sym_matrix(np.array(ets.mean(axis=0)), np.ones(n_roi))

    gm = GradientMaps(
        n_components=1,
        approach=approach,
        kernel=kernel,
        random_state=random_state,
        alignment="procrustes",
    )

    avg_gradient = gm.fit(avg_conn, sparsity=sparsity)
    reference = avg_gradient.gradients_

    ets = pd.DataFrame(ets)

    tv_gradients = pd.DataFrame(np.zeros(timeseries.shape))
    tv_gradients.columns = [f"ROI {x + 1}" for x in range(n_roi)]

    for label, current_connectome in ets.iterrows():
        start = time.time()

        # nxn connectome:
        nxnconn = vec_to_sym_matrix(
            np.array(current_connectome), np.ones(n_roi)
        )

        timepoint_gradient_map = gm.fit(
            nxnconn, reference=reference, sparsity=sparsity
        )
        tv_gradients.iloc[label] = timepoint_gradient_map.aligned_[0]

        stop = time.time()
        duration = stop - start
        print(
            f"Round {label + 1}/{n_timepoints} took {duration} seconds!",
            end="\r",
        )

    return tv_gradients
