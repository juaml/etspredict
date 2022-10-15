#!/usr/bin/env python3

import numpy as np
import pandas as pd

import etspredict.timeseries as gts


def test_get_edge_timeseries():
    timeseries = np.random.rand(100, 10)
    ets = gts.get_edge_timeseries(timeseries)
    assert ets.shape == (100, 45)


def test_global_measures():
    timeseries = np.random.rand(5, 3)
    gm = gts.global_measures(timeseries)
    assert isinstance(gm, pd.DataFrame)
    assert gm.shape == (5, 9)


def test_principal_gradient():
    timeseries = np.random.rand(3, 300)
    grad = gts.principal_gradient(timeseries)
    assert isinstance(grad, pd.DataFrame)
    assert grad.shape == timeseries.shape
