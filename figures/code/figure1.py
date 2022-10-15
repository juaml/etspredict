import os
from itertools import combinations

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

# output
outfile = os.path.join(
    "..", "images", "paper", f"Figure1_Identification_{dataset}"
)

# plotting parameters

# heatmap

cmap_name_ident = "autumn"  # "hot"#"OrRd"
cmap_name_idiff = "autumn"  # "hot"#"OrRd"

vmin_idiff, vmax_idiff = 0, 25
vmin_ident, vmax_ident = 0, 100

vmin_idiff_heatmap, vmax_idiff_heatmap = 0, 25
vmin_ident_heatmap, vmax_ident_heatmap = 50, 100

grid_rows, grid_cols = 2, 5
# hspace, wspace = 0.35, 0.6
width_ratios = (0.20, 0.20, 0.02, 0.56, 0.02)


cm = 1 / 2.54
figsize = (18 * cm, 10 * cm)


##############################################################################


def load_transform_results():

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


def prepare_heatmap(df_combined_bins):

    df_combined_bins_cp = df_combined_bins.copy()
    idiff_matrix = np.zeros((20, 20))
    ident_acc_matrix = np.zeros((20, 20))

    # loop over bin combinations, plot one in upper triangle, in in lower
    # triangle
    for bin1, bin2 in combinations(range(1, 21), 2):
        bin1_mask = df_combined_bins_cp["bin1"] == bin1
        combined_bin_row = df_combined_bins_cp.loc[bin1_mask]
        bin2_mask = combined_bin_row["bin2"] == bin2
        combined_bin_row = combined_bin_row.loc[bin2_mask]

        idiff_matrix[bin2 - 1, bin1 - 1] = combined_bin_row["idiff"]
        ident_acc_matrix[bin1 - 1, bin2 - 1] = combined_bin_row[
            "identification_accuracy"
        ]

    # package as dataframes for seaborn
    heatmap_idiff = pd.DataFrame(idiff_matrix)
    heatmap_idiff.index = [f"Bin {x}" for x in range(1, 21)]
    heatmap_idiff.columns = [f"Bin {x}" for x in range(1, 21)]

    heatmap_ident_acc = pd.DataFrame(ident_acc_matrix)
    heatmap_ident_acc.index = [f"Bin {x}" for x in range(1, 21)]
    heatmap_ident_acc.columns = [f"Bin {x}" for x in range(1, 21)]

    return heatmap_idiff, heatmap_ident_acc


def plot_A_sequential(df_sequential, ax):
    df_sequential = df_sequential.copy()
    sns.lineplot(
        ax=ax,
        data=df_sequential,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Differential Identifiability",
        x="Threshold (%)",
        palette=["r", "b"],
        markers=True,
        markeredgecolor="k",
        markeredgewidth=0.3,
        ci="sd",
        dashes=False
        # linewidth=10,
        # markersize=2
    )
    # despine
    ax.get_legend().remove()

    return ax


def plot_B_sequential(df_sequential, ax):
    df_sequential = df_sequential.copy()

    sns.lineplot(
        ax=ax,
        data=df_sequential,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Identification Accuracy (%)",
        x="Threshold (%)",
        palette=["r", "b"],
        dashes=False,
        markers=True,
        markeredgecolor="k",
        markeredgewidth=0.3,
        ci="sd",
    )
    for line in ax.legend(
        bbox_to_anchor=(0.5, 0.5), borderaxespad=0, ncol=1, fontsize=6
    ).get_lines():
        line.set_linewidth(1.5)
        line.set_markersize(3)
        line.set_markeredgewidth(0.3)

    return ax


def plot_C_individual(df_individual_bins, ax):
    df_individual_bins = df_individual_bins.copy()

    stand_pal = sns.color_palette("bwr", n_colors=20)[::-1]
    plot_ind = sns.barplot(
        data=df_individual_bins,
        x="Co-fluctuation Bin",
        y="Differential Identifiability",
        ax=ax,
        palette=stand_pal,
        linewidth=0,
    )
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)

    ax.set_xticks(ax.get_xticks()[::19])
    ax.set_ylim(vmin_idiff, vmax_idiff)

    return ax


def plot_D_individual(df_individual_bins, ax):
    df_individual_bins = df_individual_bins.copy()
    stand_pal = sns.color_palette("bwr", n_colors=20)[::-1]
    plot_ind = sns.barplot(
        data=df_individual_bins,
        x="Co-fluctuation Bin",
        y="Identification Accuracy (%)",
        ax=ax,
        palette=stand_pal,
        linewidth=0,
    )
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)
    ax.set_xticks(ax.get_xticks()[::19])
    ax.set_ylim(vmin_ident, vmax_ident)

    return ax


def plot_E_combined(heatmap_idiff, heatmap_ident, ax):
    heatmap_idiff, heatmap_ident = heatmap_idiff.copy(), heatmap_ident.copy()

    # prepare masks so the irrelevant triangle in each heatmap can be hidden
    triu_inds = np.triu_indices(20, k=0)
    mask_idiff = np.zeros((20, 20), dtype=bool)
    mask_idiff[triu_inds] = True

    tril_inds = np.tril_indices(20, k=0)
    mask_ident_acc = np.zeros((20, 20), dtype=bool)
    mask_ident_acc[tril_inds] = True

    plot = sns.heatmap(
        heatmap_idiff,
        ax=ax,
        vmin=vmin_idiff_heatmap,
        vmax=vmax_idiff_heatmap,
        cmap=cmap_name_idiff,
        cbar=False,
        annot=False,
        mask=mask_idiff,
    )
    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    plot2 = sns.heatmap(
        heatmap_ident,
        ax=ax,
        vmin=vmin_ident_heatmap,
        vmax=vmax_ident_heatmap,
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
    norm_idiff = mpl.colors.Normalize(vmin=vmin_idiff, vmax=vmax_idiff)
    cb_idiff = mpl.colorbar.ColorbarBase(
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
    norm_ident = mpl.colors.Normalize(vmin=50, vmax=vmax_ident)
    cb_ident = mpl.colorbar.ColorbarBase(
        ax_ident,
        cmap=cmap_ident,
        norm=norm_ident,
        orientation="vertical",
        ticklocation="right",
    )
    cb_ident.set_label("Identification Accuracy (%)", rotation=270)

    return ax_idiff, ax_ident


def plot_scatter_C_D(df_individual_bins):

    plot = sns.scatterplot(
        data=df_individual_bins,
        x="Identification Accuracy (%)",
        y="Differential Identifiability",
    )
    return plot


def figure_one():

    (
        df_individual_bins,
        df_combined_bins,
        df_sequential,
    ) = load_transform_results()
    heatmap_idiff, heatmap_ident_acc = prepare_heatmap(
        df_combined_bins=df_combined_bins
    )

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)

        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,  # hspace=hspace,
            width_ratios=width_ratios,  # wspace=wspace,
        )

        # sequential idiff
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_A_sequential(df_sequential, ax=ax1)

        # sequential identification accuracy
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_B_sequential(df_sequential, ax=ax2)

        # individual bins idiff
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("C", fontweight="bold", loc="left", fontsize=10)
        ax3 = plot_C_individual(df_individual_bins, ax=ax3)

        # individual bins ident acc
        ax4 = fig.add_subplot(grid[1, 1])
        ax4.set_title("D", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_D_individual(df_individual_bins, ax=ax4)

        # heatmap identification v idiff
        ax5 = fig.add_subplot(grid[0:4, 3])
        ax5.set_title("E", fontweight="bold", loc="left", fontsize=10)
        ax5 = plot_E_combined(heatmap_idiff, heatmap_ident_acc, ax=ax5)

        # heatmap colorbars
        ax6, ax7 = fig.add_subplot(grid[0:4, 2]), fig.add_subplot(grid[:, 4])
        ax6, ax7 = plot_E_colorbars(ax6, ax7)

        # save images
        # plt.savefig(f"{outfile}.png", dpi=300)

        sns.despine(fig, top=True, right=True)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)
        plt.close()

    plot_scatter_C_D(df_individual_bins)
    plt.savefig(f"{outfile}_scatterplot.svg")


if __name__ == "__main__":
    figure_one()
