#!/usr/bin/env python3

import os
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

##############################################################################
# Constants


dataset = "hcp_aging"
parcellation = "schaefer200x17"


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

idiff_path_sequential = os.path.join(
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
    "..", "images", "paper", f"Figure5_Identification_{dataset}"
)

# figure parameters
grid_row, grid_col = 2, 5
# hspace, wspace = 0.35, 0.6
width_ratios = (0.20, 0.20, 0.02, 0.56, 0.02)

vmin_idiff, vmax_idiff = 0, 30
vmin_ident, vmax_ident = 50, 100

cmap_name_ident = "autumn"  # "OrRd"
cmap_name_idiff = "autumn"  # "OrRd"

cm = 1 / 2.54
figsize = (18 * cm, 10 * cm)


##############################################################################


def load_transform_results():

    bin_dict = {}
    for i in range(1, 9):
        bin_dict[f"Rank {i}"] = f"Bin {i}"

    idiff_combined_bins = pd.read_csv(idiff_path_combined_bins, sep="\t")
    idiff_combined_bins["identification_accuracy"] = (
        (
            idiff_combined_bins["ISR REST1-REST2"]
            + idiff_combined_bins["ISR REST2-REST1"]
        )
        / 2
    ) * 100
    # idiff_combined_bins["idiff"] = idiff_combined_bins["idiff"] / 100

    idiff_individual_bins = pd.read_csv(idiff_path_individual_bins, sep="\t")

    # idiff_individual_bins["idiff"] = idiff_individual_bins["idiff"] / 100

    idiff_individual_bins["identification_accuracy"] = (
        (
            idiff_individual_bins["ISR REST1-REST2"]
            + idiff_individual_bins["ISR REST2-REST1"]
        )
        / 2
    ) * 100

    idiff_individual_bins = idiff_individual_bins.rename(
        {
            "identification_accuracy": "Identification Accuracy (%)",
            "idiff": "Differential Identifiability",
            "threshold": "Threshold (%)",
            "bin": "Co-fluctuation bin",
        },
        axis="columns",
    )
    idiff_individual_bins["Co-fluctuation bin"] = idiff_individual_bins[
        "Co-fluctuation bin"
    ].replace(bin_dict)

    idiff_sequential = pd.read_csv(idiff_path_sequential, sep="\t")
    idiff_sequential = idiff_sequential.rename(
        columns={"amplitude magnitude": "Co-fluctuation level"}
    )

    # idiff_sequential["idiff"] = idiff_sequential["idiff"] / 100

    idiff_sequential["identification_accuracy"] = (
        (
            idiff_sequential["identification_accuracy_r2r1"]
            + idiff_sequential["identification_accuracy_r1r2"]
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
    ].replace({"high": "HACF", "low": "LACF"})

    return idiff_individual_bins, idiff_combined_bins, idiff_sequential


def prepare_heatmaps(idiff_combined_bins):

    idiff_matrix = np.zeros((8, 8))
    ident_acc_matrix = np.zeros((8, 8))

    for bin1, bin2 in combinations(range(1, 9), 2):

        bin1_mask = idiff_combined_bins["bin1"] == bin1
        combined_bin_row = idiff_combined_bins.loc[bin1_mask]
        bin2_mask = combined_bin_row["bin2"] == bin2
        combined_bin_row = combined_bin_row.loc[bin2_mask]

        idiff_matrix[bin2 - 1, bin1 - 1] = combined_bin_row["idiff"]
        ident_acc_matrix[bin1 - 1, bin2 - 1] = combined_bin_row[
            "identification_accuracy"
        ]

    heatmap_idiff = pd.DataFrame(idiff_matrix)
    heatmap_idiff.index = range(1, 9)
    heatmap_idiff.columns = range(1, 9)

    heatmap_ident_acc = pd.DataFrame(ident_acc_matrix)
    heatmap_ident_acc.index = range(1, 9)
    heatmap_ident_acc.columns = range(1, 9)

    triu_inds = np.triu_indices(8, k=0)
    mask_idiff = np.zeros((8, 8), dtype=bool)
    mask_idiff[triu_inds] = True

    tril_inds = np.tril_indices(8, k=0)
    mask_ident_acc = np.zeros((8, 8), dtype=bool)
    mask_ident_acc[tril_inds] = True

    heatmap_idiff.columns = [f"Bin {x}" for x in heatmap_idiff.columns]
    heatmap_idiff.index = [f"Bin {x}" for x in heatmap_idiff.index]
    heatmap_ident_acc.columns = [f"Bin {x}" for x in heatmap_ident_acc.columns]
    heatmap_ident_acc.index = [f"Bin {x}" for x in heatmap_ident_acc.index]

    return heatmap_idiff, heatmap_ident_acc, mask_idiff, mask_ident_acc


def plot_A_sequential_idiff(idiff_sequential, ax):

    sns.lineplot(
        ax=ax,
        data=idiff_sequential,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Differential Identifiability",
        x="Threshold (%)",
        palette=["r", "b"],
        markers=True,
        ci=None,
        dashes=False,
        linewidth=1.5,
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.3,
    )

    ax.get_legend().remove()

    return ax


def plot_B_sequential_ident(idiff_sequential, ax):

    sns.lineplot(
        ax=ax,
        data=idiff_sequential,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Identification Accuracy (%)",
        x="Threshold (%)",
        palette=["r", "b"],
        dashes=False,
        markers=True,
        ci=None,
        linewidth=1.5,
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.3,
    )

    legend = ax.legend(
        bbox_to_anchor=(0.45, 0.45),
        loc="lower left",
        borderaxespad=0,
        fontsize=6,
    )
    for line in legend.get_lines():
        line.set_linewidth(1.5)
        line.set_markersize(3)

    return ax


def plot_C_individual_idiff(idiff_individual_bins, ax):

    stand_pal = sns.color_palette("bwr", n_colors=9)[::-1]
    plot_ind = sns.barplot(
        data=idiff_individual_bins,
        x="Co-fluctuation bin",
        y="Differential Identifiability",
        ax=ax,
        palette=stand_pal,
    )
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)

    return ax


def plot_D_individual_ident(idiff_individual_bins, ax):

    stand_pal = sns.color_palette("bwr", n_colors=9)[::-1]

    plot_ind = sns.barplot(
        data=idiff_individual_bins,
        x="Co-fluctuation bin",
        y="Identification Accuracy (%)",
        ax=ax,
        palette=stand_pal,
    )
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)

    return ax


def plot_E_combined_bins(idiff_combined_bins, ax):
    (
        heatmap_idiff,
        heatmap_ident_acc,
        mask_idiff,
        mask_ident_acc,
    ) = prepare_heatmaps(idiff_combined_bins)

    plot = sns.heatmap(
        heatmap_idiff,
        ax=ax,
        vmin=vmin_idiff,
        vmax=vmax_idiff,
        cmap=cmap_name_idiff,
        cbar=False,
        annot=False,
        mask=mask_idiff,
    )

    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    plot2 = sns.heatmap(
        heatmap_ident_acc,
        ax=ax,
        vmin=vmin_ident,
        vmax=vmax_ident,
        cmap=cmap_name_ident,
        cbar=False,
        annot=False,
        mask=mask_ident_acc,
    )

    plot2.set_xticklabels(plot2.get_xticklabels(), rotation=60)
    plot2.set_yticklabels(plot2.get_yticklabels(), rotation=0)

    return ax


def plot_E_colorbars(ax_idiff, ax_ident):
    cmap_idiff = plt.get_cmap(cmap_name_idiff)
    norm_idiff = matplotlib.colors.Normalize(vmin=vmin_idiff, vmax=vmax_idiff)
    cb_idiff = matplotlib.colorbar.ColorbarBase(
        ax_idiff,
        cmap=cmap_idiff,
        norm=norm_idiff,
        orientation="vertical",
        ticklocation="left",
    )
    cb_idiff.set_label(
        "Differential Identifiability",
    )

    cmap_ident = plt.get_cmap(cmap_name_ident)
    norm_ident = matplotlib.colors.Normalize(vmin=vmin_ident, vmax=vmax_ident)
    cb_ident = matplotlib.colorbar.ColorbarBase(
        ax_ident,
        cmap=cmap_ident,
        norm=norm_ident,
        orientation="vertical",
        ticklocation="right",
    )
    cb_ident.set_label("Identification Accuracy (%)", rotation=270)

    return ax_idiff, ax_ident


def figure_five():

    (
        idiff_individual_bins,
        idiff_combined_bins,
        idiff_sequential,
    ) = load_transform_results()

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_row,
            grid_col,  # hspace=hspace,
            width_ratios=width_ratios,  # wspace=wspace,
        )

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_A_sequential_idiff(idiff_sequential, ax1)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_B_sequential_ident(idiff_sequential, ax2)

        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("C", fontweight="bold", loc="left", fontsize=10)
        ax3 = plot_C_individual_idiff(idiff_individual_bins, ax3)

        ax4 = fig.add_subplot(grid[1, 1])
        ax4.set_title("D", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_D_individual_ident(idiff_individual_bins, ax4)

        ax5 = fig.add_subplot(grid[:, 3])
        ax5.set_title("E", fontweight="bold", loc="left", fontsize=10)
        ax5 = plot_E_combined_bins(idiff_combined_bins, ax5)

        # heatmap colorbars
        ax6, ax7 = fig.add_subplot(grid[:, 2]), fig.add_subplot(grid[:, 4])
        ax6, ax7 = plot_E_colorbars(ax6, ax7)

        # save images
        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    figure_five()
