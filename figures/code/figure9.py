import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from baycomp import two_on_single

from etspredict.prediction.scoring_utils import compare_propabilites

###############################################################################
# paths to results where sc had been removed
###############################################################################


sc_free_individual_path = (
    f"../../results/intermediate/sc_free_predictions/"
    "individual_bins/hcp_individual_bins_kernelridge_scikit_schaefer200x17.csv"
)

sc_free_combined_path = (
    f"../../results/intermediate/sc_free_predictions/"
    "combined_bins/hcp_combined_bins_kernelridge_scikit_schaefer200x17.csv"
)

sc_free_sequential_path = (
    f"../../results/intermediate/sc_free_predictions/"
    "sequential/hcp_sequential_kernelridge_scikit_schaefer200x17.csv"
)


##############################################################################
# paths to results where sc had not been removed
##############################################################################


individual_path = (
    f"../../results/intermediate/predictions/"
    "individual_bins/hcp_individual_bins_kernelridge_scikit_schaefer200x17.csv"
)

combined_path = (
    f"../../results/intermediate/predictions/"
    "combined_bins/hcp_combined_bins_kernelridge_scikit_schaefer200x17.csv"
)

sequential_path = (
    f"../../results/intermediate/predictions/"
    "sequential/hcp_sequential_kernelridge_scikit_schaefer200x17.csv"
)


###############################################################################
#
###############################################################################


vmin = 0
vmax = 0.6
cmap_name = "autumn"
unique_bins = ["1", "5", "10", "15", "20"]
# width_ratios=(0.33, 0.33, 0.33, 0.02)
height_ratios = (0.02, 0.44, 0.44)
grid_rows, grid_cols = 3, 3
cm = 1 / 2.54
figsize = (18 * cm, 11 * cm)
rope_perc = 0.05

# outfile
outfile = os.path.join(
    "..", "images", "paper", f"Figure9_schaefer200x17_behavioural_sc_free"
)


def load_transform_results():

    target_dict = {
        "ReadEng_Unadj": "Reading (pronounciation)",
        "PMAT24_A_CR": "Fluid Intelligence (PMAT)",
        "PicVocab_Unadj": "Vocabulary (picture matching)",
    }
    individual_df_sc_free = pd.read_csv(sc_free_individual_path)
    combined_df_sc_free = pd.read_csv(sc_free_combined_path)
    sequential_df_sc_free = pd.read_csv(sc_free_sequential_path)

    individual_df_sc_free["target"] = individual_df_sc_free["target"].replace(
        target_dict
    )

    combined_df_sc_free["target"] = combined_df_sc_free["target"].replace(
        target_dict
    )

    sequential_df_sc_free["target"] = sequential_df_sc_free["target"].replace(
        target_dict
    )

    sequential_df_sc_free = sequential_df_sc_free.rename(
        columns={
            "threshold": "Threshold (in %)",
            "test_corr": "Pearson's r",
            "rank": "Co-fluctuation level",
        }
    )
    sequential_df_sc_free["Co-fluctuation level"] = sequential_df_sc_free[
        "Co-fluctuation level"
    ].replace({"high": "HACF", "low": "LACF"})
    return individual_df_sc_free, combined_df_sc_free, sequential_df_sc_free


def plot_panel_labels(fig):
    fig.text(
        0.02,
        0.92,
        "A",
        ha="center",
        va="center",
        rotation=0,
        size=10,
        fontweight="bold",
    )

    fig.text(
        0.02,
        0.47,
        "B",
        ha="center",
        va="center",
        rotation=0,
        size=10,
        fontweight="bold",
    )
    return fig


def prepare_heatmaps(individual_df_sc_free, combined_df_sc_free, target):

    #######################################################################
    # get individual bins ready for diagonal and full fc for upper corner
    #######################################################################

    target_mask_ind_sc_free = individual_df_sc_free["target"] == target
    individual_df_target_subset_sc_free = individual_df_sc_free.loc[
        target_mask_ind_sc_free
    ]

    hm_diagonal_sc_free = []
    heatmap_sc_free = np.zeros((5, 5))
    annotation_matrix = np.empty((5, 5), dtype="<U4")

    full_fc_mask = individual_df_target_subset_sc_free["rank"] == "full"
    full_fc_scores = individual_df_target_subset_sc_free.loc[full_fc_mask][
        "test_corr"
    ]
    annotations_list = []
    for u_bin in unique_bins:
        # sc free
        u_bin_mask = individual_df_target_subset_sc_free["rank"] == u_bin
        individual_df_target_bin_subset_sc_free = (
            individual_df_target_subset_sc_free.loc[u_bin_mask]
        )
        bin_scores = individual_df_target_bin_subset_sc_free["test_corr"]

        hm_diagonal_sc_free.append(bin_scores.mean())

        # do rope analysis
        p_left, p_rope, p_right = two_on_single(
            bin_scores.values, full_fc_scores.values, rope_perc
        )
        comparison = compare_propabilites(p_left, p_rope, p_right)
        annotations_list.append(comparison)

    np.fill_diagonal(heatmap_sc_free, hm_diagonal_sc_free)
    np.fill_diagonal(annotation_matrix, annotations_list)

    heatmap_sc_free[0, 4] = full_fc_scores.mean()

    #######################################################################
    # combined bins
    #######################################################################

    target_mask_comb_sc_free = combined_df_sc_free["target"] == target
    combined_df_target_subset_sc_free = combined_df_sc_free.loc[
        target_mask_comb_sc_free
    ]

    for i_bin1, bin1 in enumerate(unique_bins):
        for i_bin2, bin2 in enumerate(unique_bins):
            rank = "_".join([bin1, bin2])
            if rank in list(combined_df_sc_free["rank"]):

                combined_bin_mask_sc_free = (
                    combined_df_target_subset_sc_free["rank"] == rank
                )
                combined_bin_subset_sc_free = (
                    combined_df_target_subset_sc_free.loc[
                        combined_bin_mask_sc_free
                    ]
                )

                scores_combined_bin = combined_bin_subset_sc_free["test_corr"]

                # do rope analysis
                p_left, p_rope, p_right = two_on_single(
                    scores_combined_bin.values,
                    full_fc_scores.values,
                    rope_perc,
                )
                comparison = compare_propabilites(p_left, p_rope, p_right)
                annotation_matrix[i_bin2, i_bin1] = comparison

                heatmap_sc_free[i_bin2, i_bin1] = scores_combined_bin.mean()

    heatmap_df_sc_free = pd.DataFrame(heatmap_sc_free)
    heatmap_df_sc_free.columns = [f"Bin {x}" for x in unique_bins]
    heatmap_df_sc_free.index = [f"Bin {x}" for x in unique_bins]

    annotation_matrix[0, 4] = round(full_fc_scores.mean(), 2)

    return heatmap_df_sc_free, annotation_matrix


def plot_A_ind_combined_behaviour(
    individual_df_sc_free, combined_df_sc_free, target, ax
):

    # indices of upper triangle, heatmap mask
    triu_inds = np.triu_indices(5, k=1)
    heatmap_mask = np.zeros((5, 5), dtype=bool)
    heatmap_mask[triu_inds] = True

    # leave a space at the top right for full connectome
    heatmap_mask[0, 4] = False

    heatmap_df, annotation_matrix = prepare_heatmaps(
        individual_df_sc_free, combined_df_sc_free, target
    )
    plot = sns.heatmap(
        heatmap_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_name,
        cbar=False,
        annot=annotation_matrix,
        mask=heatmap_mask,
        annot_kws={"fontsize": 10},
        fmt="",
    )
    ax.text(
        2.5,
        0.5,
        "Full Connectome:",
        ha="center",
        va="center",
        rotation=0,
        size=7,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="gainsboro", ec="k", lw=2),
    )
    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    return ax


def plot_B_sequential(target_seq, ax):

    sns.lineplot(
        ax=ax,
        data=target_seq,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Pearson's r",
        x="Threshold (in %)",
        palette=["r", "b"],
        ci="sd",
        markers=True,
        markeredgewidth=0.3,
        markeredgecolor="k",
    )
    ax.set_ylim(vmin, vmax)
    ax.legend(bbox_to_anchor=(0.95, 0.55), borderaxespad=0)

    return ax


def plot_colorbar(ax):

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", ticklocation="top"
    )
    cbar.set_label("Pearson's r")

    return ax


def figure_nine():

    (
        individual_df_sc_free,
        combined_df_sc_free,
        sequential_df_sc_free,
    ) = load_transform_results()
    targets = individual_df_sc_free.target.unique()[::-1]

    with plt.style.context("style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,
            height_ratios=height_ratios,
            # height_ratios=(0.32, 0.02, 0.32, 0.02, 0.32),
        )

        fig = plot_panel_labels(fig)
        ax_color = fig.add_subplot(grid[0, :])
        ax_color = plot_colorbar(ax_color)

        for i_target, target in enumerate(targets):

            fig_row = 1
            fig_col = i_target

            ax1 = fig.add_subplot(grid[fig_row, fig_col])

            if target == "Reading (pronounciation)":
                target_label = "Reading (pronunciation)"
            else:
                target_label = target
            ax1.set_title(f"{target_label}")
            ax1 = plot_A_ind_combined_behaviour(
                individual_df_sc_free, combined_df_sc_free, target, ax1
            )
            if fig_col != 0:
                ax1.set_yticks([])

            fig_row = 2
            target_mask = sequential_df_sc_free["target"] == target
            target_seq = sequential_df_sc_free.loc[target_mask]
            ax2 = fig.add_subplot(grid[fig_row, fig_col])
            ax2 = plot_B_sequential(target_seq, ax2)
            if fig_col != 3:
                ax2.get_legend().remove()

        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    figure_nine()
