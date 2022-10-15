import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

##############################################################################
# Constants


dataset = "hcp"
parcellation = "schaefer200x17"

idiff_path_combined_bins = os.path.join(
    "..",
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
    "..",
    "results",
    "idiff_sequential",
    dataset,
    "GS",
    f"dataset-{dataset}_preprocessing_level-GS_parcellation-{parcellation}"
    "_criterion-rss.tsv",
)

# output
outfile = os.path.join(
    "..", "..", "images", "suppl", f"Identification_{dataset}_suppl_sequential"
)

# plotting parameters

# heatmap

cmap_name_ident = "autumn"  # "hot"#"OrRd"
cmap_name_idiff = "autumn"  # "hot"#"OrRd"

vmin_idiff, vmax_idiff = 0, 25
vmin_ident, vmax_ident = 0, 100

grid_rows, grid_cols = 1, 2

cm = 1 / 2.54
figsize = (18 * cm, 10 * cm)
cmap_scatter = "bwr"


##############################################################################


def load_transform_results():

    # load and prepare all result data
    # combined bins
    idiff_combined_bins = pd.read_csv(idiff_path_combined_bins, sep="\t")
    idiff_combined_bins["Identification Accuracy (%)"] = (
        (
            idiff_combined_bins["ISR REST1-REST2"]
            + idiff_combined_bins["ISR REST2-REST1"]
        )
        / 2
    ) * 100
    idiff_combined_bins["Differential Identifiability"] = idiff_combined_bins[
        "idiff"
    ]
    bin_names = []
    for i, j in zip(idiff_combined_bins["bin1"], idiff_combined_bins["bin2"]):
        k = " ".join([str(i), str(j)])
        bin_names.append(f"Bins {k}")
    idiff_combined_bins["Co-fluctuation Bin"] = bin_names

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


def plot_correlations(df_sequential, ax):

    hacf_df_mask = df_sequential["Co-fluctuation level"] == "HACF"
    lacf_df_mask = df_sequential["Co-fluctuation level"] == "LACF"

    hacf_df = df_sequential.loc[hacf_df_mask]
    lacf_df = df_sequential.loc[lacf_df_mask]

    ax.fill_between(
        x=hacf_df["Threshold (%)"],
        y1=hacf_df["mean within subjects"],
        y2=hacf_df["mean between subjects"],
        color="r",
        alpha=0.3,
        label="HACF",
    )

    ax.fill_between(
        x=lacf_df["Threshold (%)"],
        y1=lacf_df["mean within subjects"],
        y2=lacf_df["mean between subjects"],
        color="b",
        alpha=0.3,
        label="LACF",
    )

    plt.ylabel("Pearson's r")
    plt.xlabel("Threshold (%)")
    ax.legend(loc="lower right", fontsize=6)

    return ax


def plot_derivatives(df_sequential, ax):

    hacf_df_mask = df_sequential["Co-fluctuation level"] == "HACF"
    lacf_df_mask = df_sequential["Co-fluctuation level"] == "LACF"

    hacf_df = df_sequential.loc[hacf_df_mask]
    lacf_df = df_sequential.loc[lacf_df_mask]

    hacf_df["mean within"] = np.insert(
        np.diff(hacf_df["mean within subjects"]), 0, np.nan
    )
    hacf_df["mean between"] = np.insert(
        np.diff(hacf_df["mean between subjects"]), 0, np.nan
    )

    lacf_df["mean within"] = np.insert(
        np.diff(lacf_df["mean within subjects"]), 0, np.nan
    )
    lacf_df["mean between"] = np.insert(
        np.diff(lacf_df["mean between subjects"]), 0, np.nan
    )

    plot_data = pd.concat([hacf_df, lacf_df], axis=0).reset_index()[
        [
            "Threshold (%)",
            "Co-fluctuation level",
            "mean within",
            "mean between",
        ]
    ]
    plot_data = plot_data.melt(
        id_vars=["Co-fluctuation level", "Threshold (%)"],
        value_vars=["mean within", "mean between"],
        var_name="between/within",
        value_name="rate of change",
    )

    id_list = []
    for i, j in zip(
        plot_data["Co-fluctuation level"], plot_data["between/within"]
    ):
        id_list.append(" - ".join([i, j]))

    plot_data["id"] = id_list

    sns.lineplot(
        data=plot_data,
        x="Threshold (%)",
        y="rate of change",
        hue="id",
        style="id",
        palette=[
            "r",
            "b",
            "lightcoral",
            "cornflowerblue",
        ],
        markers=True,
        linewidth=1.5,
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.3,
        dashes=False,
    )
    ax.legend(bbox_to_anchor=(0.5, 0.5), borderaxespad=0, ncol=1, fontsize=6)

    return ax


def figure_suppl_identification():

    (
        df_individual_bins,
        df_combined_bins,
        df_sequential,
    ) = load_transform_results()

    with plt.style.context("../style.mplstyle"):
        fig = plt.figure(figsize=figsize)

        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,  # hspace=hspace,
        )
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_correlations(df_sequential, ax1)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_derivatives(df_sequential, ax2)

        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)
        plt.savefig(f"{outfile}.png", dpi=400)


if __name__ == "__main__":
    figure_suppl_identification()
