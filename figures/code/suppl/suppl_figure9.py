import os

import matplotlib as mpl
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
    "..",
    "..",
    "images",
    "suppl",
    f"Identification_{dataset}_suppl_combined_bins",
)

# plotting parameters

# heatmap

cmap_name_ident = "autumn"  # "hot"#"OrRd"
cmap_name_idiff = "autumn"  # "hot"#"OrRd"

vmin_idiff, vmax_idiff = 0, 25
vmin_ident, vmax_ident = 0, 100

grid_rows, grid_cols = 3, 3
width_ratios = (0.19, 0.01, 0.44)
height_ratios = (0.19, 0.19, 0.19)

cm = 1 / 2.54
figsize = (18 * cm, 15 * cm)
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


def plot_scatter(df_combined_bins, ax):
    # embed()
    plot = sns.scatterplot(
        data=df_combined_bins,
        ax=ax,
        x="Identification Accuracy (%)",
        y="Differential Identifiability",
        hue="Co-fluctuation Bin",
        palette=sns.color_palette("bwr", n_colors=190)[::-1],
    )
    plot.get_legend().remove()
    return ax


def plot_scatter_cbar(ax):
    cmap = plt.get_cmap("bwr_r")
    norm = mpl.colors.Normalize(vmin=1, vmax=190)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="vertical", ticklocation="right"
    )
    cb.set_label("Co-fluctuation bin", rotation=270, labelpad=10)
    cb.set_ticks([x for x in range(1, 200, 9)])
    return ax


def plot_identification(df_combined_bins, ax):

    sns.barplot(
        data=df_combined_bins,
        x="Co-fluctuation Bin",
        y="Identification Accuracy (%)",
        palette=sns.color_palette("bwr", n_colors=190)[::-1],
    )
    ax.set_xticklabels([])
    return ax


def plot_idiff(df_combined_bins, ax):
    sns.barplot(
        data=df_combined_bins,
        x="Co-fluctuation Bin",
        y="Differential Identifiability",
        palette=sns.color_palette("bwr", n_colors=190)[::-1],
    )
    plt.xticks(rotation=60)
    for label in ax.xaxis.get_ticklabels():
        label.set_visible(False)

    for label in ax.xaxis.get_ticklabels()[::5]:
        label.set_visible(True)

    return ax


def plot_derivatives(df_combined_bins, ax):

    co_bins = df_combined_bins["Co-fluctuation Bin"]
    plot_data = pd.DataFrame()
    plot_data["Co-fluctuation Bin"] = co_bins

    mws = df_combined_bins["mean within subjects"].copy()
    mbs = df_combined_bins["mean between subjects"].copy()

    plot_data["mean within"] = np.insert(np.diff(mws), 0, 0)
    plot_data["mean between"] = np.insert(np.diff(mbs), 0, 0)
    # plot_data["Iacc"] = np.diff(i_acc)
    # plot_data["Idiff"] = np.diff(i_acc)

    plot_data = pd.melt(
        frame=plot_data,
        id_vars="Co-fluctuation Bin",
        var_name="Derivative",
        value_name="rate of change",
    )
    sns.pointplot(
        data=plot_data,
        x="Co-fluctuation Bin",
        y="rate of change",
        hue="Derivative",
        ax=ax,
        palette=["#1b9e77", "#7570b3"],
        scale=0.2,
    )
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1, 1), ncol=1, fancybox=True, fontsize=6)
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
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_scatter(df_combined_bins, ax1)

        ax_cbar = fig.add_subplot(grid[0, 1])
        ax_cbar = plot_scatter_cbar(ax_cbar)

        ax2 = fig.add_subplot(grid[1, 2])
        ax2.set_title("", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_derivatives(df_combined_bins, ax2)

        """
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("C", fontweight="bold", loc="left", fontsize=10)
        ax3 = plot_correlations(df_combined_bins, ax3)
        """

        ax4 = fig.add_subplot(grid[0, 2])
        ax4.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_identification(df_combined_bins, ax4)

        ax5 = fig.add_subplot(grid[2, 2])
        # ax5.set_title("", fontweight="bold", loc="left", fontsize=10)
        ax5 = plot_idiff(df_combined_bins, ax5)

        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)
        plt.savefig(f"{outfile}.png", dpi=400)


if __name__ == "__main__":
    figure_suppl_identification()
