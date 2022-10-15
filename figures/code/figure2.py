import os
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from baycomp import two_on_single

from etspredict.prediction.scoring_utils import compare_propabilites

##############################################################################
# Constants

dataset = "hcp"
parcellation = "schaefer200x17"
model = "kernelridge_scikit"
cbpm = False
rope_perc = 0.05
cmap_name = "autumn"
ranklist = ["1", "5", "10", "15", "20"]
ampl_rankings = ["_".join([x, y]) for x, y in combinations(ranklist, 2)]

n_targets = 9
accuracy_metric = "test_corr"

# figure parameters
grid_rows, grid_cols, width_ratios = 3, 4, (0.32, 0.32, 0.32, 0.02)
cm = 1 / 2.54
vmin, vmax = 0, 0.45
figsize = (18 * cm, 14 * cm)

##############################################################################


if cbpm:
    predictions_path = os.path.join(
        "..", "..", "results", "intermediate", "cbpm", "predictions"
    )

    # outfile
    outfile = os.path.join(
        "..",
        "images",
        "paper",
        f"{dataset}_{parcellation}_{accuracy_metric}_"
        f"combined_bins_heatmaps_cbpm_{model}",
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
        f"{dataset}_{parcellation}_{accuracy_metric}_"
        f"combined_bins_heatmaps_{model}",
    )

# individual bins connectivity
individual_bins_path = os.path.join(
    predictions_path,
    "individual_bins",
    f"{dataset}_individual_bins_{model}_{parcellation}.csv",
)

# combined bins
combined_bins_path = os.path.join(
    predictions_path,
    "combined_bins",
    f"{dataset}_combined_bins_{model}_{parcellation}.csv",
)


def select_targets(full_fc_df):
    full_fc_ordered_by_score = (
        full_fc_df.groupby("target")
        .mean()
        .sort_values(by=accuracy_metric, ascending=False)
    )

    return list(full_fc_ordered_by_score.index)[:n_targets]


def load_transform_results():
    individual_bins_df = pd.read_csv(individual_bins_path)

    # individual bins also contains data for full functional connectivity
    full_fc_mask = individual_bins_df["ampl_rankings"] == "full"
    individual_bins_mask = individual_bins_df["ampl_rankings"] != "full"

    full_fc_df = individual_bins_df.loc[full_fc_mask]
    individual_bins_df = individual_bins_df.loc[individual_bins_mask]

    # combined bins
    combined_bins_df = pd.read_csv(combined_bins_path)

    return full_fc_df, individual_bins_df, combined_bins_df


def get_heatmap_mask():
    triu_inds = np.triu_indices(5, k=1)
    heatmap_mask = np.zeros((5, 5), dtype=bool)
    heatmap_mask[triu_inds] = True

    # leave a space at the top right for full connectome
    heatmap_mask[0, 4] = False
    return heatmap_mask


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
        annotation = compare_propabilites(p_left, p_rope, p_right)

        # second prepare diagonal
        ind_bin_mean_list.append(
            target_rank_ind_bin_df[accuracy_metric].mean()
        )
        annotations_list.append(annotation)

    return ind_bin_mean_list, annotations_list


def bayesian_rope_body(combined_bins_df, target, target_full_fc_df):

    heatmap_matrix = np.zeros((5, 5))
    annotation_matrix = np.empty((5, 5), dtype="<U4")

    for i_col, column in enumerate(ranklist):
        for i_row, row in enumerate(ranklist):
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
                heatmap_matrix[i_row, i_col] = target_comb_bins_df[
                    accuracy_metric
                ].mean()

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


def prepare_heatmap_and_annotations(
    full_fc_df, individual_bins_df, combined_bins_df, target
):
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
    heatmap_matrix[0, 4] = full_fc_mean_score
    annotation_matrix[0, 4] = f"{round(full_fc_mean_score, 2)}"

    # dataframe and labels
    labels = [f"Bin {i}" for i in ranklist]
    heatmap_df = pd.DataFrame(heatmap_matrix, columns=labels, index=labels)
    annotation_df = pd.DataFrame(annotation_matrix)

    return heatmap_df, annotation_df


def plot_heatmap(
    fig, grid, grid_row, grid_col, heatmap_df, annotation_df, target
):
    ax = fig.add_subplot(grid[grid_row, grid_col])
    heatmap_mask = get_heatmap_mask()

    plot = sns.heatmap(
        heatmap_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_name,
        cbar=False,
        annot=annotation_df,
        mask=heatmap_mask,
        annot_kws={"fontsize": 10},
        fmt="",
    )
    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    if target in ["Reading (pronounciation)"]:
        target_label = "Reading (pronunciation)"
    else:
        target_label = target

    plt.title(target_label)

    # add an arrow that points at full fc score
    ax.text(
        2.5,
        0.5,
        "Full Connectome:",
        ha="center",
        va="center",
        rotation=0,
        size=7,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="gainsboro", ec="k", lw=0.5),
    )
    # remove tick labels except at edges
    if grid_col != 0:
        ax.set_yticks([])

    if grid_row != 2:
        ax.set_xticks([])


def plot_colorbar(fig, grid):
    ax = fig.add_subplot(grid[:, 3])
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")


def figure_two():

    full_fc_df, individual_bins_df, combined_bins_df = load_transform_results()
    targets = select_targets(full_fc_df)

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows, grid_cols, width_ratios=width_ratios
        )

        grid_row, grid_col = 0, 0

        for _, target in enumerate(targets):
            heatmap_df, annotation_df = prepare_heatmap_and_annotations(
                full_fc_df, individual_bins_df, combined_bins_df, target
            )

            plot_heatmap(
                fig,
                grid,
                grid_row,
                grid_col,
                heatmap_df,
                annotation_df,
                target,
            )
            # update grid indices
            if grid_col != 2:
                grid_col += 1
            else:
                grid_col = 0
                grid_row += 1

        plot_colorbar(fig, grid)
        # plt.tight_layout()

        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    figure_two()
