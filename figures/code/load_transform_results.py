import os

import pandas as pd


def load_transform_results_figure1(dataset, parcellation):

    idiff_path_combined_bins = os.path.join(
        "..",
        "..",
        "results",
        "idiff_combined_bins",
        dataset,
        "GS",
        f"dataset-{dataset}_preprocessing_level-GS_parcellation-schaefer200x17"
        "_criterion-rss.tsv",
    )

    idiff_path_individual_bins = os.path.join(
        "..",
        "..",
        "results",
        "idiff_individual_bins",
        dataset,
        "GS",
        f"dataset-{dataset}_preprocessing_level-GS_parcellation-schaefer200x17"
        "_criterion-rss.tsv",
    )

    idiff_path_sequental = os.path.join(
        "..",
        "..",
        "results",
        "idiff_sequential",
        dataset,
        "GS",
        f"dataset-{dataset}_preprocessing_level-GS_parcellation-{parcellation}"
        "_criterion-rss.tsv",
    )

    # load and prepare all result data
    # combined bins
    idiff_combined_bins = pd.read_csv(idiff_path_combined_bins, sep="\t")
    idiff_combined_bins["identification_accuracy"] = (
        (
            idiff_combined_bins["ISR REST1-REST2"]
            + idiff_combined_bins["ISR REST2-REST1"]
        )
        / 2
    ) * 100
    idiff_combined_bins["idiff"] = idiff_combined_bins["idiff"]

    # individual bins
    idiff_individual_bins = pd.read_csv(idiff_path_individual_bins, sep="\t")
    idiff_individual_bins[
        "Differential Identifiability"
    ] = idiff_individual_bins["idiff"]

    idiff_individual_bins["Identification Accuracy (%)"] = (
        (
            idiff_individual_bins["ISR REST1-REST2"]
            + idiff_individual_bins["ISR REST2-REST1"]
        )
        / 2
    ) * 100
    idiff_individual_bins["Co-fluctuation Bin"] = [
        f"Bin {x}" for x in range(1, 21)
    ]

    # sequential
    idiff_sequential = pd.read_csv(idiff_path_sequental, sep="\t")
    idiff_sequential = idiff_sequential.rename(
        columns={"amplitude magnitude": "Co-fluctuation level"}
    )
    idiff_sequential["idiff"] = idiff_sequential["idiff"]
    idiff_sequential["identification_accuracy"] = (
        (
            idiff_sequential["identification_accuracy_r1r2"]
            + idiff_sequential["identification_accuracy_r2r1"]
        )
        / 2
    ) * 100

    idiff_sequential = idiff_sequential.rename(
        {
            "identification_accuracy": "Identification Accuracy (%)",
            "idiff": "Differential Identifiability",
            "threshold": "Threshold (%)",
        },
        axis="columns",
    )

    idiff_sequential["Co-fluctuation level"] = idiff_sequential[
        "Co-fluctuation level"
    ].replace("high", "HACF")
    idiff_sequential["Co-fluctuation level"] = idiff_sequential[
        "Co-fluctuation level"
    ].replace("low", "LACF")

    return idiff_individual_bins, idiff_combined_bins, idiff_sequential
