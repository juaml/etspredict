#!/usr/bin/env python3

from itertools import combinations

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance

from etspredict.timeseries import get_edge_timeseries


def ets_amplitudes(edge_timeseries):
    """Calculates co-fluctuation amplitudes at each frame of a given edge
    timeseries. The co-fluctuation amplitudes are quantified as root sum
    squares [RSS] as outlined in
    (https://www.pnas.org/content/117/45/28393#sec-21).
    Parameters
    -----------
    edge_timeseries : np.array
        number of rows should correspond to number of time frames
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    Returns
    --------
    RSS : np.ndarray
        root sum of squares for each time frame in the edge timeseries
    """

    assert isinstance(edge_timeseries, np.ndarray)
    # power of 0.5 is same as square root

    return np.sum(edge_timeseries**2, 1) ** 0.5


def reorder_timeseries(timeseries, scores, ascending=False):
    """Reorder timeseries based on a score for each timepoint.

    Parameters
    ----------
    timeseries : np.ndarray or pd.DataFrame
        number of rows should correspond to number of time points
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    scores : np.ndarray or pd.Series
        one-dimensional array with a length that corresponds to the number of
        timeframes in the timeseries.

    Returns
    -------
    reordered_timeseries : pd.DataFrame
        Reordered timeseries with scores as index and timeseries as values.
        Rows (timepoints) are ordered descending from highest to lowest scores.
        Each column corresponds to a node.
    """

    # check that all scores are a numeric type
    scores = np.array(scores).astype(float)
    reordered_timeseries = pd.DataFrame(timeseries)
    reordered_timeseries.index = scores
    reordered_timeseries.sort_index(axis=0, ascending=ascending, inplace=True)

    return reordered_timeseries


def select_high_and_low_connectome(
    timeseries,
    threshold=5,
    criterion="rss",
):
    """Takes in a timeseries, orders timepoints by a specified criterion
    (default is "rss") and returns high- and low-magnitude timepoint-based
    connectomes.

    ----------
    timeseries : np.array
        number of rows should correspond to number of time frames
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    criterion : str or array-like
        "rss" (default) or an array with each element corresponding to a
        timepoint. Timepoints will then be reordered according to the array.
    threshold : int
        percentage frames to choose at high and low end

    """

    n_timepoints, n_roi = timeseries.shape
    ets = get_edge_timeseries(timeseries)

    if criterion in ["rss"]:
        criterion = ets_amplitudes(ets)

    reordered_timeseries = reorder_timeseries(timeseries, criterion)

    n_frames = round((threshold / 100) * n_timepoints)
    low_end = n_timepoints - n_frames
    low_chunk = reordered_timeseries.values[low_end:, :]
    high_chunk = reordered_timeseries.values[:n_frames, :]

    conn_measure = ConnectivityMeasure(
        cov_estimator=EmpiricalCovariance(),
        kind="correlation",
        vectorize=True,
        discard_diagonal=True,
    )
    low_conn = conn_measure.fit_transform([low_chunk])[0]
    high_conn = conn_measure.fit_transform([high_chunk])[0]

    connectomes = pd.DataFrame()
    connectomes["high"] = high_conn
    connectomes["low"] = low_conn

    return connectomes


def select_all_slice_connectomes(
    timeseries,
    threshold=5,
    criterion="rss",
):
    """Slice a timeseries into chunks of a given size (as a percentage of
    the whole) and get a connectome for each chunk.

    Parameters
    -----------
    timeseries : np.ndarray or pd.DataFrame
        number of rows should correspond to number of timepoints
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    threshold : int
        size of chunks to be used as a percentage. I.e. if each chunk should
        make up 5 percent of the whole slice_size should be five
    criterion : str or array-like
        "rss" (default) or an array with each element corresponding to a
        timepoint. Timepoints will then be reordered according to the array.

    Returns
    --------
    chunks : pd.DataFrame
        returns a pd.DataFrame containing the unique flattened lower triangle
        for each chunk. Rows correspond to the chunk of the timeseries,
        columns correspond to the unique edges of the (lower triangle) of the
        connectome.
    """

    # make sure timeseries is now a np.array with numeric data
    timeseries = np.array(timeseries).astype(float)

    n_timepoints, n_roi = timeseries.shape
    n_unique_edges = int(n_roi * (n_roi - 1) / 2)

    ets = get_edge_timeseries(timeseries)

    if isinstance(criterion, str):
        if criterion in ["rss"]:
            criterion = ets_amplitudes(ets)
    else:
        criterion = np.array(criterion)
        assert len(criterion) == n_timepoints, (
            "Length of custom criterion scores should match number of "
            "timepoints!"
        )

    timeseries = reorder_timeseries(timeseries, criterion).values

    # n_slices is the number of slices per chunk
    # threshold is the chunk size as a percentage of the timeseries
    # n_chunks is the number of different chunks the timeseries is partitioned
    # into
    n_slices = int(threshold / 100 * n_timepoints)
    assert (
        n_timepoints % n_slices == 0
    ), "Choose a percentage such that each frame can be used"
    n_chunks = int(n_timepoints / n_slices)
    chunks = np.zeros((n_chunks, n_unique_edges))

    previous_chunk_end = 0
    for i in range(1, n_chunks + 1):
        current_chunk_end = i * n_slices
        chunk = timeseries[previous_chunk_end:current_chunk_end, :]
        previous_chunk_end = current_chunk_end

        conn_measure = ConnectivityMeasure(
            kind="correlation",
            vectorize=True,
            discard_diagonal=True,
            cov_estimator=EmpiricalCovariance(),
        )
        connectome = conn_measure.fit_transform([chunk])[0]

        chunks[i - 1] = connectome

    chunks = pd.DataFrame(chunks).T
    chunks.columns = [f"Rank {i + 1}" for i in range(chunks.shape[1])]

    return chunks


def all_combined_bins_connectomes(timeseries, threshold=5, criterion="rss"):
    """Slice a timeseries into chunks of a given size (as a percentage of
    the whole) and get a connectome for each chunk.

    Parameters
    -----------
    timeseries : np.ndarray or pd.DataFrame
        number of rows should correspond to number of timepoints
        (first dimension), number of columns should correspond to number of
        nodes (second dimension)
    threshold : int
        size of chunks to be used as a percentage. I.e. if each chunk should
        make up 5 percent of the whole slice_size should be five
    criterion : str or array-like
        "rss" (default) or an array with each element corresponding to a
        timepoint. Timepoints will then be reordered according to the array.

    Returns
    --------
    chunks : pd.DataFrame
        returns a pd.DataFrame containing the unique flattened lower triangle
        for each chunk. Rows correspond to the chunk of the timeseries,
        columns correspond to the unique edges of the (lower triangle) of the
        connectome.
    """

    # make sure timeseries is now a np.array with numeric data
    timeseries = np.array(timeseries).astype(float)

    n_timepoints, n_roi = timeseries.shape

    ets = get_edge_timeseries(timeseries)

    if isinstance(criterion, str):
        if criterion in ["rss"]:
            criterion = ets_amplitudes(ets)
    else:
        criterion = np.array(criterion)
        assert len(criterion) == n_timepoints, (
            "Length of custom criterion scores should match number of "
            "timepoints!"
        )

    timeseries = reorder_timeseries(timeseries, criterion).values

    # n_slices is the number of slices per chunk
    # threshold is the chunk size as a percentage of the timeseries
    # n_chunks is the number of different chunks the timeseries is partitioned
    # into
    n_slices = int(threshold / 100 * n_timepoints)
    assert (
        n_timepoints % n_slices == 0
    ), "Choose a percentage such that each frame can be used"
    n_chunks = int(n_timepoints / n_slices)

    each_bin_dict = {}
    comb_bins_dict = {}
    previous_chunk_end = 0
    for i in range(1, n_chunks + 1):
        current_chunk_end = i * n_slices
        chunk = timeseries[previous_chunk_end:current_chunk_end, :]
        previous_chunk_end = current_chunk_end

        each_bin_dict[str(i)] = chunk

    for bin1, bin2 in combinations(each_bin_dict.keys(), 2):

        comb_bins = np.concatenate(
            [each_bin_dict[bin1], each_bin_dict[bin2]], axis=0
        )

        conn_measure = ConnectivityMeasure(
            kind="correlation",
            vectorize=True,
            discard_diagonal=True,
            cov_estimator=EmpiricalCovariance(),
        )
        connectome = conn_measure.fit_transform([comb_bins])[0]

        comb_bins_dict["_".join([bin1, bin2])] = connectome

    return pd.DataFrame(comb_bins_dict)
