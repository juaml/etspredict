#!/usr/bin/env python3

import os
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from baycomp import two_on_single
from tvgraphs.prediction.scoring_utils import avg_cv_scores

from etspredict.prediction.scoring_utils import compare_propabilites

##############################################################################
# Constants


dataset = "hcp_aging"
parcellation = "schaefer200x17"
model = "kernelridge_scikit"
cbpm = False
rope_perc = 0.05
cmap_name = "autumn"
n_bins = 8
accuracy_metric = "test_corr"


ranklist = [str(x) for x in range(1, n_bins + 1)]
ampl_rankings = ["_".join([x, y]) for x, y in combinations(ranklist, 2)]


# figure parameters
grid_rows, grid_cols = 3, 4
height_ratios = (0.02, 0.49, 0.49)
hspace, wspace = 0.4, 0.3
vmin, vmax = 0, 0.5

cm = 1 / 2.54
figsize = (18 * cm, 10 * cm)

target_dict = {
    "nih_fluidcogcomp_unadjusted": "Fluid Cog. Comp. Score",
    "nih_dccs_unadjusted": "Cog. Flexibility",
    "tpvt_uss": "Lang./Vocab. Comprehension",
    "nih_crycogcomp_unadjusted": "Crystal. Cog. Comp. Score",
}

if cbpm:
    predictions_path = os.path.join(
        "..", "..", "results", "intermediate", "cbpm", "predictions"
    )

    # outfile
    outfile = os.path.join(
        "..",
        "images",
        "paper",
        f"Figure6_{dataset}_combined_bins_heatmaps_cbpm_{model}",
    )

else:
    predictions_path = os.path.join(
        "..", "..", "results", "intermediate", "predictions"
    )

    # outfile
    outfile = os.path.join(
        "..",
        "images",
        "paper",
        f"Figure6_{dataset}_combined_bins_heatmaps_{model}",
    )

# combined bins
combined_bins_path = os.path.join(
    predictions_path,
    "combined_bins",
    f"{dataset}_combined_bins_{model}_{parcellation}.csv",
)


# individual bins connectivity
individual_bins_path = os.path.join(
    predictions_path,
    "individual_bins",
    f"{dataset}_individual_bins_{model}_{parcellation}.csv",
)


def select_targets(full_fc_df):
    full_fc_ordered_by_score = (
        full_fc_df.groupby("target")
        .mean()
        .sort_values(by="test_corr", ascending=False)
    )

    return full_fc_ordered_by_score.index


def load_transform_results():
    # individual bins load data
    individual_bins_df = pd.read_csv(individual_bins_path)

    # individual bins also contains data for full functional connectivity
    full_fc_mask = individual_bins_df["ampl_rankings"] == "full"
    individual_bins_mask = individual_bins_df["ampl_rankings"] != "full"

    full_fc_df = individual_bins_df.loc[full_fc_mask]
    individual_bins_df = individual_bins_df.loc[individual_bins_mask]

    # combined bins
    combined_bins_df = pd.read_csv(combined_bins_path)

    # sequential
    sequential_path = os.path.join(
        predictions_path,
        "sequential",
        f"{dataset}_sequential_{model}_{parcellation}.csv",
    )
    sequential_df = pd.read_csv(sequential_path)

    return full_fc_df, individual_bins_df, combined_bins_df, sequential_df


def get_heatmap_mask():
    # indices of upper triangle, heatmap mask
    triu_inds = np.triu_indices(n_bins, k=1)
    heatmap_mask = np.zeros((n_bins, n_bins), dtype=bool)
    heatmap_mask[triu_inds] = True

    # leave a space at the top right for full connectome
    heatmap_mask[0, n_bins - 1] = False

    return heatmap_mask


def prepare_sequential(sequential_df, target):
    target_mask_seq = sequential_df["target"] == target
    seq_target_df = sequential_df.loc[target_mask_seq]

    high_seq_target_df = seq_target_df.loc[
        seq_target_df["ampl_rankings"] == "high"
    ]
    low_seq_target_df = seq_target_df.loc[
        seq_target_df["ampl_rankings"] == "low"
    ]

    sequential_to_plot = {
        "Co-fluctuation level": [],
        "Threshold (in %)": [],
        "Pearson's r": [],
    }
    # thresholds
    for thresh in seq_target_df["threshold"].unique():

        # HACF
        thresh_high_seq_target_df = high_seq_target_df.loc[
            high_seq_target_df["threshold"] == thresh
        ]
        cv_scores = np.mean(
            np.vstack(
                thresh_high_seq_target_df[accuracy_metric].values.reshape(5, 5)
            ),
            axis=1,
        )
        sequential_to_plot["Co-fluctuation level"] += [
            "HACF" for _ in range(len(cv_scores))
        ]
        sequential_to_plot["Threshold (in %)"] += [
            thresh for _ in range(len(cv_scores))
        ]
        sequential_to_plot["Pearson's r"] += list(cv_scores)

        # LACF
        thresh_low_seq_target_df = low_seq_target_df.loc[
            low_seq_target_df["threshold"] == thresh
        ]
        cv_scores = np.mean(
            np.vstack(
                thresh_low_seq_target_df[accuracy_metric].values.reshape(5, 5)
            ),
            axis=1,
        )
        sequential_to_plot["Co-fluctuation level"] += [
            "LACF" for _ in range(len(cv_scores))
        ]
        sequential_to_plot["Threshold (in %)"] += [
            thresh for _ in range(len(cv_scores))
        ]
        sequential_to_plot["Pearson's r"] += list(cv_scores)

    return pd.DataFrame(sequential_to_plot)


def bayesian_rope_diagonal(individual_bins_df, target, target_full_fc_df):

    # target specific individual bins results
    target_ind_bins_mask = individual_bins_df["target"] == target
    target_ind_bins_df = individual_bins_df.loc[target_ind_bins_mask]

    # rope for every rank, add each rank score to diagonal array
    ind_bin_mean_list = []
    annotations_list = []
    for rank in ranklist:

        rank_mask_ind_bin = target_ind_bins_df["ampl_rankings"] == rank
        target_rank_ind_bin_df = target_ind_bins_df.loc[rank_mask_ind_bin]

        # first do rope analysis
        p_left, p_rope, p_right = two_on_single(
            target_rank_ind_bin_df[accuracy_metric].values,
            target_full_fc_df[accuracy_metric].values,
            rope_perc,
        )
        comparison = compare_propabilites(p_left, p_rope, p_right)

        # second prepare diagonal
        avg_cv = avg_cv_scores(
            np.vstack(target_rank_ind_bin_df[accuracy_metric])
        )[0]

        ind_bin_mean_list.append(avg_cv)
        annotations_list.append(comparison)

    return ind_bin_mean_list, annotations_list


def bayesian_rope_body(combined_bins_df, target, target_full_fc_df):
    heatmap_matrix = np.zeros((n_bins, n_bins))
    annotation_matrix = np.empty((n_bins, n_bins), dtype="<U4")

    for i_col, column in enumerate(ranklist):
        for i_row, row in enumerate(ranklist):

            # only use unique half of matrix
            rank = "_".join([column, row])
            if rank in ampl_rankings:

                target_comb_bins_mask = (
                    combined_bins_df["ampl_rankings"] == rank
                ) & (combined_bins_df["target"] == target)

                target_comb_bins_df = combined_bins_df.loc[
                    target_comb_bins_mask
                ]

                # assign mean score for target ~ combined bin FC to
                # appropriate place in the heatmap matrix
                avg_cv = avg_cv_scores(
                    np.vstack(target_comb_bins_df[accuracy_metric])
                )[0]
                heatmap_matrix[i_row, i_col] = avg_cv

                # apply bayesian rope comparison
                p_left, p_rope, p_right = two_on_single(
                    target_comb_bins_df[accuracy_metric].values,
                    target_full_fc_df[accuracy_metric].values,
                    rope_perc,
                )
                # and assign as label for heatmap
                annotation_matrix[i_row, i_col] = compare_propabilites(
                    p_left, p_rope, p_right
                )

    return heatmap_matrix, annotation_matrix


def prepare_heatmaps(full_fc_df, individual_bins_df, combined_bins_df, target):
    # target specific full fc results
    target_full_fc_mask = full_fc_df["target"] == target
    target_full_fc_df = full_fc_df.loc[target_full_fc_mask]

    # initialise heatmap and annotations
    diagonal_values, diagonal_annotations = bayesian_rope_diagonal(
        individual_bins_df, target, target_full_fc_df
    )
    heatmap_matrix, annotation_matrix = bayesian_rope_body(
        combined_bins_df, target, target_full_fc_df
    )

    # format heatmap and annotations
    np.fill_diagonal(heatmap_matrix, diagonal_values)
    np.fill_diagonal(annotation_matrix, diagonal_annotations)

    # add full fc score
    full_fc_mean_score = target_full_fc_df[accuracy_metric].mean()
    heatmap_matrix[0, n_bins - 1] = full_fc_mean_score
    annotation_matrix[0, n_bins - 1] = f"{round(full_fc_mean_score, 2)}"

    # dataframe and labels
    labels = [f"Bin {i}" for i in ranklist]
    heatmap_df = pd.DataFrame(heatmap_matrix, columns=labels, index=labels)
    annotation_df = pd.DataFrame(annotation_matrix)

    return heatmap_df, annotation_df


def plot_colorbar(ax):

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="horizontal", ticklocation="top"
    )
    cbar.set_label("Pearson's r", fontsize=7)

    return ax


def plot_sequential(target_seq, ax):

    sns.lineplot(
        ax=ax,
        data=target_seq,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Pearson's r",
        x="Threshold (in %)",
        ci="sd",
        palette=["r", "b"],
        markers=True,
        linewidth=1.5,
        markersize=3,
        markeredgecolor="k",
        dashes=False,
        markeredgewidth=0.3,
    )
    ax.set_ylim(vmin, vmax)
    ax.legend(bbox_to_anchor=(0.95, 0.55), borderaxespad=0, fontsize=6)

    return ax


def plot_heatmap(heatmap_df, annotation_df, ax):
    heatmap_mask = get_heatmap_mask()
    plot = sns.heatmap(
        heatmap_df.round(2),
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_name,
        cbar=False,
        annot=annotation_df,
        mask=heatmap_mask,
        annot_kws={"fontsize": 5},
        fmt="",
    )

    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    ax.text(
        3.9,
        0.5,
        "Full Connectome:",
        ha="center",
        va="center",
        rotation=0,
        size=5,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="gainsboro", ec="k", lw=1),
    )

    return ax


def figure_six():

    (
        full_fc_df,
        individual_bins_df,
        combined_bins_df,
        sequential_df,
    ) = load_transform_results()
    targets = select_targets(full_fc_df)

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,
            height_ratios=height_ratios,
            # hspace=hspace, wspace=wspace
        )

        grid_row, grid_col = 1, 0

        for target in targets:
            heatmap_df, annotation_df = prepare_heatmaps(
                full_fc_df, individual_bins_df, combined_bins_df, target
            )

            # heatmaps
            ax_hm = fig.add_subplot(grid[grid_row, grid_col])
            ax_hm.set_title(target_dict[target])
            ax_hm = plot_heatmap(heatmap_df, annotation_df, ax_hm)
            if grid_col != 0:
                ax_hm.set_yticks([])

            # remove tick labels except at edges

            # sequential plot
            target_seq = prepare_sequential(sequential_df, target)

            ax_seq = fig.add_subplot(grid[grid_row + 1, grid_col])
            ax_seq = plot_sequential(target_seq, ax_seq)
            if grid_col != 3:
                ax_seq.get_legend().remove()

            # update grid
            grid_col += 1

        ax_color = fig.add_subplot(grid[0, :])
        ax_color = plot_colorbar(ax_color)

        fig.text(
            0.01,
            0.9,
            "A",
            ha="center",
            va="center",
            rotation=0,
            size=10,
            fontweight="bold",
        )

        fig.text(
            0.01,
            0.45,
            "B",
            ha="center",
            va="center",
            rotation=0,
            size=10,
            fontweight="bold",
        )

        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    figure_six()
