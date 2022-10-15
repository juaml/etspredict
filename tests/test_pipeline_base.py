import os
import shutil

import numpy as np
import pandas as pd

from etspredict import selective_connectomes as sc
from etspredict import timeseries as gts
from etspredict.pipelines.base import Pipeline


def test_pipeline():

    # pretend to load subject specific data
    subject = "151425"
    run = "REST1LR"
    timeseries = np.random.rand(100, 10) + 10

    functions = {
        gts.global_measures: [],
        gts.principal_gradient: ["gaussian", "pca"],
        sc.select_high_and_low_connectome: [10, "rss"],
        sc.select_all_slice_connectomes: [5],
        sc.all_combined_bins_connectomes: [5],
    }
    for func, vals in functions.items():

        out_dir = "test_data"
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            print(f"{out_dir} already exists!")

        outpath_elements = [subject, run]
        outfile_parameters = {
            "subject": subject,
            "function": func.__name__,
        }
        for i, param in enumerate(vals):
            if isinstance(param, str) or isinstance(param, int):
                outfile_parameters[f"arg{i}"] = param

        assert out_dir is not None, "out_dir is None"

        pipeline = Pipeline(
            timeseries=timeseries,
            root_dir=out_dir,
            path_elements=outpath_elements,
            file_parameters=outfile_parameters,
            save_index=False,
        )

        result = pipeline.run_wrapped_function(func, *vals)

        assert os.path.isdir(pipeline.output_directory)
        assert os.path.isfile(pipeline.outfile)

        test_load = pd.read_csv(pipeline.outfile, sep="\t")

        np.testing.assert_almost_equal(test_load.values, result.values)

        shutil.rmtree(out_dir)
