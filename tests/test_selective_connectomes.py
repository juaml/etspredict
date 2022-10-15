#!/usr/bin/env python3

from itertools import product

import numpy as np
import pandas as pd

import etspredict.selective_connectomes as sc


def test_ets_amplitudes():
    ets = np.arange(9).reshape(3, 3)
    ampl = sc.ets_amplitudes(ets)
    assert len(ampl) == 3


def test_reorder_timeseries():
    # test custom criterion

    timeseries = np.random.rand(10, 300)
    scores = np.arange(10)
    np.random.shuffle(scores)
    reord = sc.reorder_timeseries(timeseries, scores)
    assert np.array_equal(reord.index.values, np.flip(np.arange(10)))

    # test ascending = True
    reord = sc.reorder_timeseries(timeseries, scores, ascending=True)
    assert np.array_equal(reord.index.values, np.arange(10))


def test_high_and_low_connectome():
    timeseries = np.random.rand(100, 300)

    thresholds = [x for x in range(5, 55, 5)]
    criterions = ["rss", np.arange(100)]

    for threshold, criterion in product(thresholds, criterions):

        # test rss option
        connectomes = sc.select_high_and_low_connectome(
            timeseries=timeseries,
            criterion=criterion,
            threshold=threshold,
        )

        assert isinstance(connectomes, pd.DataFrame)
        assert connectomes.shape == (44850, 2)
        reference = connectomes.iloc[:, 0]
        columns_bools = []
        for col in connectomes:
            columns_bools.append(reference.equals(connectomes[col]))

        assert not all(columns_bools), "All columns are the same"


def test_select_all_slice_connectomes():
    timeseries = np.random.rand(100, 300)

    criterions = ["rss", np.arange(100)]
    chunk_sizes = [5, 10, 20, 25]

    for chunk_size, criterion in product(chunk_sizes, criterions):

        connectomes = sc.select_all_slice_connectomes(
            timeseries=timeseries, criterion=criterion, threshold=chunk_size
        )

        shape = (44850, int(100 / chunk_size))
        assert isinstance(connectomes, pd.DataFrame)
        assert connectomes.shape == shape
        reference = connectomes.iloc[:, 0]
        columns_bools = []
        for col in connectomes:
            columns_bools.append(reference.equals(connectomes[col]))

        assert not all(columns_bools), "All columns are the same"


def test_all_combined_bins_connectomes():
    timeseries = np.random.rand(100, 300)
    criterions = ["rss", np.arange(100)]
    chunk_sizes = [5, 10, 20, 25]

    for chunk_size, criterion in product(chunk_sizes, criterions):

        connectomes = sc.all_combined_bins_connectomes(
            timeseries=timeseries, criterion=criterion, threshold=chunk_size
        )
        term = 100 / chunk_size
        dim1 = int(term * (term - 1) / 2)
        shape = (44850, dim1)
        assert isinstance(connectomes, pd.DataFrame)
        assert connectomes.shape == shape
        reference = connectomes.iloc[:, 0]
        columns_bools = []
        for col in connectomes:
            columns_bools.append(reference.equals(connectomes[col]))

        assert not all(columns_bools), "All columns are the same"
