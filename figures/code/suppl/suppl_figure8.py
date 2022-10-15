import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_transform_results import (
    load_idiff_matrices_results,
    load_transform_results_suppl_8,
)

##############################################################################
# Constants


dataset = "hcp"
parcellation = "schaefer200x17"


# output
outfile = os.path.join(
    "..",
    "..",
    "images",
    "suppl",
    f"Identification_{dataset}_suppl_individual_bins",
)

# plotting parameters

# heatmap

cmap_name_ident = "autumn"  # "hot"#"OrRd"
cmap_name_idiff = "autumn"  # "hot"#"OrRd"

vmin_idiff, vmax_idiff = 0, 25
vmin_ident, vmax_ident = 0, 100

grid_rows, grid_cols = 4, 3
width_ratios = (0.19, 0.01, 0.44)
height_ratios = (0.19, 0.19, 0.19, 0.19)

cm = 1 / 2.54
figsize = (18 * cm, 15 * cm)
cmap_scatter = "bwr"


##############################################################################


def plot_scatter(df_individual_bins, ax):

    plot = sns.scatterplot(
        data=df_individual_bins,
        ax=ax,
        x="Identification Accuracy (%)",
        y="Differential Identifiability",
        hue="Co-fluctuation Bin",
        palette=sns.color_palette("bwr", n_colors=20)[::-1],
    )
    plot.get_legend().remove()
    return ax


def plot_scatter_cbar(ax):
    cmap = plt.get_cmap("bwr_r")
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="vertical", ticklocation="right"
    )
    cb.set_label("Co-fluctuation bin", rotation=270, labelpad=10)
    cb.set_ticks([x for x in range(1, 21, 2)])
    return ax


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


def plot_identification(df_individual_bins, ax):

    sns.barplot(
        data=df_individual_bins,
        x="Co-fluctuation Bin",
        y="Identification Accuracy (%)",
        palette=sns.color_palette("bwr", n_colors=20)[::-1],
    )
    ax.set_xticklabels([])
    return ax


def plot_idiff(df_individual_bins, ax):
    sns.barplot(
        data=df_individual_bins,
        x="Co-fluctuation Bin",
        y="Differential Identifiability",
        palette=sns.color_palette("bwr", n_colors=20)[::-1],
    )
    plt.xticks(rotation=60)
    return ax


def plot_derivatives(df_individual_bins, ax):

    co_bins = df_individual_bins["Co-fluctuation Bin"]
    plot_data = pd.DataFrame()
    plot_data["Co-fluctuation Bin"] = co_bins

    mws = df_individual_bins["mean within subjects"].copy()
    mbs = df_individual_bins["mean between subjects"].copy()

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
        scale=0.5,
    )
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1, 1), ncol=1, fancybox=True, fontsize=6)

    # for label in ax.xaxis.get_ticklabels()[::2]:
    #    label.set_visible(False)
    return ax


def plot_violins(df, ax):

    sns.violinplot(
        data=df, x="Co-fluctuation Bin", y="Pearson's r", hue="within/between"
    )
    return ax


def figure_suppl_identification():

    idiff_matrix_results = load_idiff_matrices_results()

    (
        df_individual_bins,
        df_combined_bins,
        df_sequential,
    ) = load_transform_results_suppl_8(dataset, parcellation)
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
        ax1 = plot_scatter(df_individual_bins, ax1)

        ax_cbar = fig.add_subplot(grid[0, 1])
        ax_cbar = plot_scatter_cbar(ax_cbar)

        ax2 = fig.add_subplot(grid[1, 2])
        ax2.set_title("", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_derivatives(df_individual_bins, ax2)

        """
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("C", fontweight="bold", loc="left", fontsize=10)
        ax3 = plot_correlations(df_sequential, ax3)
        """

        ax4 = fig.add_subplot(grid[0, 2])
        ax4.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_identification(df_individual_bins, ax4)

        ax5 = fig.add_subplot(grid[2, 2])
        # ax5.set_title("", fontweight="bold", loc="left", fontsize=10)
        ax5 = plot_idiff(df_individual_bins, ax5)

        ax6 = fig.add_subplot(grid[3, 2])
        ax6 = plot_violins(idiff_matrix_results, ax6)

        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)
        plt.savefig(f"{outfile}.png", dpi=400)


if __name__ == "__main__":
    figure_suppl_identification()
