import os
from itertools import product

import numpy as np
import pandas as pd
import pkg_resources


def load_subject_list(dataset):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    stream = pkg_resources.resource_stream(__name__, path)

    return list(np.loadtxt(stream, dtype=str))


def iterate_product_and_correlate(var="fd"):

    subjects = load_subject_list("hcp")
    sessions = [
        "_".join([a, b]) for a, b in product(["REST1", "REST2"], ["LR", "RL"])
    ]
    corr_methods = ["spearman", "pearson"]
    parcellations = ["schaefer200x17", "schaefer300x17", "schaefer400x17"]
    preproc = ["GS", "no_GS"]

    results = {
        "correlation_method": [],
        "correlation": [],
        "subject": [],
        "session": [],
        "parcellation": [],
        "preproc": [],
    }

    for subject, session, parcellation, corr_method, prepr in product(
        subjects, sessions, parcellations, corr_methods, preproc
    ):

        try:
            with open(
                f"../results/{var}_rss_correlations/"
                f"{subject}_{session}_{parcellation}"
                f"_{prepr}_{corr_method}.txt",
                "r",
            ) as f:
                correlation = float(f.read())

                results["correlation"].append(correlation)
                results["subject"].append(subject)
                results["session"].append(session)
                results["parcellation"].append(parcellation)
                results["correlation_method"].append(corr_method)
                results["preproc"].append(prepr)

        except FileNotFoundError:
            print(subject, session, parcellation, corr_method)

    return pd.DataFrame(results)


def main():
    print("Doing FD")
    results_df = iterate_product_and_correlate(var="fd")
    results_df.to_csv(
        "../results/intermediate/FD_RSS_CORRELATIONS.csv", index=False
    )
    print("Doing GS")
    results_df = iterate_product_and_correlate(var="gs")
    results_df.to_csv(
        "../results/intermediate/GS_RSS_CORRELATIONS.csv", index=False
    )


if __name__ == "__main__":
    main()
