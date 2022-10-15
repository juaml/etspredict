#!/usr/bin/env python3

import os
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -------------------------------------------------------------------------- #
# Prepre input paths


dataset = "hcp"
model = "kernelridge_scikit"
n_unique_bins = 5

acc_measure = "test_r2"
font_scale = 1
vmin = 0
vmax = 0.3
cmap_name = "autumn"

dpi = 400
cm = 1 / 2.54

if acc_measure == "test_corr":
    acc_measure1 = "Pearson's r"
elif acc_measure == "test_r2":
    acc_measure1 = "r-squared"


##############################################################################
##############################################################################


def load_transform_results(parcellation, preprocessing):

    individual_bins_path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        "predictions",
        "individual_bins",
        f"{dataset}_individual_bins_{model}_"
        f"{parcellation}_{preprocessing}.csv",
    )
    combined_bins_path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        "predictions",
        "combined_bins",
        f"{dataset}_combined_bins_{model}_{parcellation}_{preprocessing}.csv",
    )
    sequential_path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        "predictions",
        "sequential",
        f"{dataset}_sequential_{model}_{parcellation}_{preprocessing}.csv",
    )
    individual_df = pd.read_csv(individual_bins_path)
    combined_df = pd.read_csv(combined_bins_path)
    sequential_df = pd.read_csv(sequential_path)

    sequential_df = sequential_df.rename(
        columns={
            "ampl_rankings": "Co-fluctuation level",
            "test_corr": "Pearson's r",
            "threshold": "Threshold (in %)",
            "test_r2": "r-squared",
        }
    )
    sequential_df["Co-fluctuation level"] = sequential_df[
        "Co-fluctuation level"
    ].replace("high", "HACF")

    sequential_df["Co-fluctuation level"] = sequential_df[
        "Co-fluctuation level"
    ].replace("low", "LACF")

    return individual_df, combined_df, sequential_df


def prepare_heatmaps(individual_df, combined_df):

    # create mask for heatmaps
    triu_inds = np.triu_indices(n_unique_bins, k=1)
    mask = np.zeros((n_unique_bins, n_unique_bins), dtype=bool)
    mask[triu_inds] = True
    mask[0, n_unique_bins - 1] = False

    # create target specific heatmaps using individual and combined bins
    target_heatmaps = {}
    for target in individual_df.target.unique():
        target_heatmaps[target] = np.zeros((n_unique_bins, n_unique_bins))

        # select target specific data to preparre individual bins for diagonal
        target_mask = individual_df["target"] == target
        individual_df_target = individual_df.loc[target_mask]

        # exclude full static FC for now
        not_full_mask = individual_df_target["ampl_rankings"] != "full"
        individual_df_target_exclude_full = individual_df_target.loc[
            not_full_mask
        ]

        full_fc_mask = individual_df_target["ampl_rankings"] == "full"
        full_fc_scores = individual_df_target.loc[full_fc_mask]

        target_heatmaps[target][0, n_unique_bins - 1] = full_fc_scores[
            acc_measure
        ].mean()

        # for cof_bin in sorted(
        #    individual_df_target_exclude_full["ampl_rankings"].unique().astype(int)
        # ):

        avg_acc_bin = individual_df_target_exclude_full.groupby(
            "ampl_rankings"
        ).mean()

        avg_acc_bin.index = avg_acc_bin.index.astype(int)
        avg_acc_bin = avg_acc_bin.sort_index()
        unique_bins_in_order = avg_acc_bin.index.astype(str)

        np.fill_diagonal(target_heatmaps[target], avg_acc_bin[acc_measure])

        # prepare combined bins for rest of heatmap
        target_mask = combined_df["target"] == target
        combined_df_target = combined_df.loc[target_mask]

        avg_acc_bin = combined_df_target.groupby("ampl_rankings").mean()

        for comb_bin, row_data in avg_acc_bin.iterrows():
            j, i = comb_bin.split("_")

            i_ind = [x for x, y in enumerate(unique_bins_in_order) if y == i][
                0
            ]
            j_ind = [x for x, y in enumerate(unique_bins_in_order) if y == j][
                0
            ]

            target_heatmaps[target][i_ind, j_ind] = row_data[acc_measure]

    return target_heatmaps, mask, unique_bins_in_order


###############################################################################
#
###############################################################################


def make_figures(parcellation, preprocessing):

    outfile = (
        f"../../images/suppl/{dataset}_{preprocessing}"
        f"_{parcellation}_{model}_{acc_measure}.pdf"
    )
    with plt.style.context("../style.mplstyle"):

        individual_df, combined_df, sequential_df = load_transform_results(
            parcellation, preprocessing
        )
        target_heatmaps, mask, unique_bins_in_order = prepare_heatmaps(
            individual_df, combined_df
        )

        fig = plt.figure(figsize=(18 * cm, 10 * cm))
        grid = fig.add_gridspec(2, 4, width_ratios=(0.3, 0.3, 0.3, 0.02))

        for col_ind, target in enumerate(individual_df.target.unique()):
            ax1 = fig.add_subplot(grid[0, col_ind])

            plot_data = pd.DataFrame(target_heatmaps[target])
            plot_data.index = [f"Bin {x}" for x in unique_bins_in_order]
            plot_data.columns = [f"Bin {x}" for x in unique_bins_in_order]

            plot = sns.heatmap(
                plot_data.round(2),
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_name,
                cbar=False,
                annot=True,
                mask=mask,
                annot_kws={"fontsize": 10},
            )
            ax1.set_title(target)
            # add an arrow that points at full fc score
            ax1.text(
                2.5,
                0.5,
                "Full Connectome:",
                ha="center",
                va="center",
                rotation=0,
                size=6,
                bbox=dict(
                    boxstyle="rarrow,pad=0.3", fc="gainsboro", ec="k", lw=1
                ),
            )
            plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
            plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

            # plot results for sequential paradigm

            target_mask_seq = sequential_df["target"] == target
            plot_data_seq = sequential_df.loc[target_mask_seq]

            ax2 = fig.add_subplot(grid[1, col_ind])

            sns.lineplot(
                ax=ax2,
                data=plot_data_seq,
                hue="Co-fluctuation level",
                style="Co-fluctuation level",
                y=acc_measure1,
                x="Threshold (in %)",
                ci="sd",
                palette=["r", "b"],
                markers=True,
                dashes=False,
                markeredgewidth=0.3,
                markeredgecolor="k",
            )
            ax2.set_ylim(vmin, vmax)
            if col_ind != 2:
                ax2.get_legend().remove()
            else:
                ax2.legend(
                    bbox_to_anchor=(0.8, 0.3), borderaxespad=0, fontsize=6
                )

        #######################################################################

        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax2 = fig.add_subplot(grid[0, 3])
        cb1 = matplotlib.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            orientation="vertical",
            ticklocation="right",
        )
        cb1.set_label(acc_measure1)

        plt.savefig(outfile, dpi=dpi)


if __name__ == "__main__":

    parcellation = ["schaefer300x17", "schaefer400x17"]
    preprocessing = ["no_GS", "GS"]
    for parc, prop in product(parcellation, preprocessing):
        make_figures(parc, prop)

    parcellation = ["schaefer200x17"]
    preprocessing = ["no_GS"]
    for parc, prop in product(parcellation, preprocessing):
        make_figures(parc, prop)
