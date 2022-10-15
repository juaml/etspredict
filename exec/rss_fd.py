import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from etspredict.prepare_data.prepare_connectomes import load_timeseries
from etspredict.selective_connectomes import ets_amplitudes
from etspredict.timeseries import get_edge_timeseries


def get_subject_rss(subject, session, parcellation, preproc="GS"):
    timeseries = load_timeseries(
        "hcp", subject, parcellation, session, preproc
    )
    ets = get_edge_timeseries(timeseries)

    return ets_amplitudes(ets)


def get_subject_confound(subject, session, fieldname):

    aml_dir = os.path.join(
        "/data", "group", "appliedml", "data", "HCP1200_Confounds_tsv_withFD"
    )
    confounds_file = os.path.join(
        f"{subject}",
        "MNINonLinear",
        "Results",
        f"rfMRI_{session}",
        f"Confounds_{subject}.tsv",
    )
    fd_file = os.path.join(aml_dir, confounds_file)

    return np.array(pd.read_csv(fd_file, sep="\t")[fieldname])


def main(subject, session, parcellation, correlation_method, preproc="GS"):

    fd = get_subject_confound(subject, session, fieldname="FD")
    rss = get_subject_rss(subject, session, parcellation, preproc=preproc)
    gs = get_subject_confound(subject, session, fieldname="GS")

    if correlation_method in ["pearson"]:
        correlation_fd, _ = pearsonr(rss, fd)
        correlation_gs, _ = pearsonr(rss, gs)
    elif correlation_method in ["spearman"]:
        correlation_fd, _ = spearmanr(rss, fd)
        correlation_gs, _ = spearmanr(rss, gs)
    else:
        raise NotImplementedError(f"{correlation_method} not implemented")

    print("correlation done")
    with open(
        f"../results/fd_rss_correlations/"
        f"{subject}_{session}_{parcellation}"
        f"_{preproc}_{correlation_method}.txt",
        "w",
    ) as f:
        f.write(f"{correlation_fd}")

    with open(
        f"../results/gs_rss_correlations/"
        f"{subject}_{session}_{parcellation}"
        f"_{preproc}_{correlation_method}.txt",
        "w",
    ) as f:
        f.write(f"{correlation_gs}")
    print("files saved")


if __name__ == "__main__":

    main(
        subject=sys.argv[1],
        session=sys.argv[2],
        parcellation=sys.argv[3],
        correlation_method=sys.argv[4],
        preproc="GS",
    )

    main(
        subject=sys.argv[1],
        session=sys.argv[2],
        parcellation=sys.argv[3],
        correlation_method=sys.argv[4],
        preproc="no_GS",
    )
