#!/usr/bin/env python3


import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from baycomp import two_on_single

from etspredict.prediction.scoring_utils import compare_propabilites

##############################################################################
# Constants


dataset = "hcp_aging"
atlas = "schaefer200x17"
sex_pred_model = "linear_svm"

cmap_name_age = "autumn"
cmap_name_sex = "autumn"

age_metric = "r-squared"

vmin_sex, vmax_sex = 50, 80
vmin_age, vmax_age = 0, 0.7

grid_rows, grid_cols = 2, 5

width_ratios = (0.20, 0.20, 0.02, 0.56, 0.02)

cm = 1 / 2.54
figsize = (18 * cm, 10 * cm)

# output
outfile = os.path.join(
    "..",
    "..",
    "images",
    "suppl",
    f"Prediction_AgeSex{dataset}_{sex_pred_model}",
)

rope_perc = 0.05
rope_perc_sex = 5

# age predictions
path_age_prediction = os.path.join(
    "..", "..", "..", "results", "intermediate", "age_predictions"
)

individual_bins_path_age = os.path.join(
    path_age_prediction,
    "individual_bins",
    f"age_prediction_{dataset}_individual_bins_kernelridge_scikit_{atlas}.csv",
)

combined_bins_path_age = os.path.join(
    path_age_prediction,
    "combined_bins",
    f"age_prediction_{dataset}_combined_bins_kernelridge_scikit_{atlas}.csv",
)

sequential_path_age = os.path.join(
    path_age_prediction,
    "sequential",
    f"age_prediction_{dataset}_sequential_kernelridge_scikit_{atlas}.csv",
)

# sex predictions
path_sex_prediction = os.path.join(
    "..", "..", "..", "results", "intermediate", "sex_predictions"
)

combined_bins_path_sex = os.path.join(
    path_sex_prediction,
    "combined_bins",
    f"sex_prediction_{dataset}_combined_bins_{sex_pred_model}_{atlas}.csv",
)
sequential_path_sex = os.path.join(
    path_sex_prediction,
    "sequential",
    f"sex_prediction_{dataset}_sequential_{sex_pred_model}_{atlas}.csv",
)
individual_path_sex = os.path.join(
    path_sex_prediction,
    "individual_bins",
    f"sex_prediction_{dataset}_individual_bins_{sex_pred_model}_{atlas}.csv",
)


##############################################################################


def load_transform_results():

    bin_dict = {}
    for i in range(1, 21):
        bin_dict[str(i)] = f"Bin {i}"

    # age prediction
    individual_bins_df_age = pd.read_csv(individual_bins_path_age)
    individual_bins_df_age = individual_bins_df_age.rename(
        {
            "test_corr": "Pearson's r",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation bin",
            "test_r2": "r-squared",
        },
        axis="columns",
    )
    individual_bins_df_age["Co-fluctuation bin"] = individual_bins_df_age[
        "Co-fluctuation bin"
    ].replace({"high": "HACF", "low": "LACF"})
    individual_bins_df_age["Co-fluctuation bin"] = individual_bins_df_age[
        "Co-fluctuation bin"
    ].replace(bin_dict)
    individual_bins_df_age["MAE"] = (
        individual_bins_df_age["test_neg_mean_absolute_error"] * -1
    )

    combined_bins_df_age = pd.read_csv(combined_bins_path_age)
    combined_bins_df_age["MAE"] = (
        combined_bins_df_age["test_neg_mean_absolute_error"] * -1
    )
    combined_bins_df_age = combined_bins_df_age.rename(
        {
            "test_corr": "Pearson's r",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation bin",
            "test_r2": "r-squared",
        },
        axis="columns",
    )
    sequential_df_age = pd.read_csv(sequential_path_age)
    sequential_df_age["MAE"] = (
        sequential_df_age["test_neg_mean_absolute_error"] * -1
    )
    sequential_df_age = sequential_df_age.rename(
        {
            "test_corr": "Pearson's r",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation level",
            "test_r2": "r-squared",
        },
        axis="columns",
    )
    sequential_df_age["Co-fluctuation level"] = sequential_df_age[
        "Co-fluctuation level"
    ].replace({"high": "HACF", "low": "LACF"})

    # sex prediction
    individual_df_sex = pd.read_csv(individual_path_sex)
    individual_df_sex = individual_df_sex.rename(
        {
            "test_balanced_accuracy": "Balanced Accuracy (%)",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation bin",
        },
        axis="columns",
    )
    individual_df_sex["Co-fluctuation bin"] = individual_df_sex[
        "Co-fluctuation bin"
    ].replace({"high": "HACF", "low": "LACF"})
    individual_df_sex["Co-fluctuation bin"] = individual_df_sex[
        "Co-fluctuation bin"
    ].replace(bin_dict)
    individual_df_sex["Balanced Accuracy (%)"] = (
        individual_df_sex["Balanced Accuracy (%)"] * 100
    )

    combined_bins_df_sex = pd.read_csv(combined_bins_path_sex)
    combined_bins_df_sex["test_balanced_accuracy"] = (
        combined_bins_df_sex["test_balanced_accuracy"] * 100
    )

    sequential_df_sex = pd.read_csv(sequential_path_sex)

    sequential_df_sex = sequential_df_sex.rename(
        {
            "test_balanced_accuracy": "Balanced Accuracy (%)",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation level",
        },
        axis="columns",
    )
    sequential_df_sex["Balanced Accuracy (%)"] = (
        sequential_df_sex["Balanced Accuracy (%)"] * 100
    )
    sequential_df_sex["Co-fluctuation level"] = sequential_df_sex[
        "Co-fluctuation level"
    ].replace({"high": "HACF", "low": "LACF"})

    # full fc
    mask_full_age = individual_bins_df_age["Co-fluctuation bin"] == "full"
    full_age = individual_bins_df_age.loc[mask_full_age]

    mask_full_sex = individual_df_sex["Co-fluctuation bin"] == "full"
    full_sex = individual_df_sex.loc[mask_full_sex]

    mask_no_full_sex = individual_df_sex["Co-fluctuation bin"] != "full"
    mask_no_full_age = individual_bins_df_age["Co-fluctuation bin"] != "full"

    individual_df_sex = individual_df_sex.loc[mask_no_full_sex]
    individual_bins_df_age = individual_bins_df_age.loc[mask_no_full_age]

    return (
        individual_bins_df_age,
        combined_bins_df_age,
        sequential_df_age,
        full_age,
        individual_df_sex,
        combined_bins_df_sex,
        sequential_df_sex,
        full_sex,
    )


def prepare_heatmap_and_annotations(
    combined_bins_age_df, combined_bins_sex_df, full_age, full_sex
):
    age_heatmap = np.zeros((8, 8))
    sex_heatmap = np.zeros((8, 8))

    age_annot = np.empty((8, 8), dtype="<U4")
    sex_annot = np.empty((8, 8), dtype="<U4")
    ampl_rankings_age = set(combined_bins_age_df["Co-fluctuation bin"])
    ampl_rankings_sex = set(combined_bins_sex_df["ampl_rankings"])

    for rank in ampl_rankings_age:
        age_mask = combined_bins_age_df["Co-fluctuation bin"] == rank
        rank_df = combined_bins_age_df.loc[age_mask]
        avg_score_age_rank = rank_df[age_metric].mean()
        idx1, idx2 = rank.split("_")
        age_heatmap[int(idx2) - 1, int(idx1) - 1] = avg_score_age_rank

        # first do rope analysis
        p_left, p_rope, p_right = two_on_single(
            rank_df[age_metric].values, full_age[age_metric].values, rope_perc
        )
        comparison = compare_propabilites(p_left, p_rope, p_right)
        age_annot[int(idx2) - 1, int(idx1) - 1] = comparison

    for rank in ampl_rankings_sex:
        sex_mask = combined_bins_sex_df["ampl_rankings"] == rank
        rank_df = combined_bins_sex_df.loc[sex_mask]
        avg_score_sex_rank = rank_df["test_balanced_accuracy"].mean()
        idx1, idx2 = rank.split("_")
        sex_heatmap[int(idx1) - 1, int(idx2) - 1] = avg_score_sex_rank

        # first do rope analysis
        p_left, p_rope, p_right = two_on_single(
            rank_df["test_balanced_accuracy"].values,
            full_sex["Balanced Accuracy (%)"].values,
            rope_perc_sex,
        )
        comparison = compare_propabilites(p_left, p_rope, p_right)
        sex_annot[int(idx1) - 1, int(idx2) - 1] = comparison

    bins = [f"Bin {x}" for x in range(1, 9)]
    age_heatmap, sex_heatmap, age_annot, sex_annot = (
        pd.DataFrame(age_heatmap, index=bins, columns=bins),
        pd.DataFrame(sex_heatmap, index=bins, columns=bins),
        pd.DataFrame(age_annot, index=bins, columns=bins),
        pd.DataFrame(sex_annot, index=bins, columns=bins),
    )

    return age_heatmap, sex_heatmap, age_annot, sex_annot


def plot_A_age_sequential(sequential_df_age, ax):

    sns.lineplot(
        ax=ax,
        data=sequential_df_age,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y=age_metric,
        ci="sd",
        x="Threshold (%)",
        palette=["r", "b"],
        markers=True,
        markeredgewidth=0.3,
        markeredgecolor="k",
        dashes=False,
    )
    ax.set_ylim(vmin_age, vmax_age)
    ax.get_legend().remove()

    return ax


def plot_B_sex_sequential(sequential_df_sex, ax):

    sns.lineplot(
        ax=ax,
        data=sequential_df_sex,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Balanced Accuracy (%)",
        x="Threshold (%)",
        ci="sd",
        palette=["r", "b"],
        markers=True,
        markeredgewidth=0.3,
        markeredgecolor="k",
        dashes=False,
    )
    ax.set_ylim(vmin_sex, vmax_sex)

    for line in ax.legend(bbox_to_anchor=(0.5, 0.5), fontsize=6).get_lines():
        line.set_linewidth(1.5)
        line.set_markersize(3)

    return ax


def plot_C_age_individual(individual_bins_df_age, ax):

    stand_pal = sns.color_palette("bwr", n_colors=8)[::-1]

    plot_ind = sns.barplot(
        data=individual_bins_df_age,
        x="Co-fluctuation bin",
        y=age_metric,
        ax=ax,
        palette=stand_pal,
        ci="sd",
        errwidth=0.4,
        linewidth=0,
    )
    ax.set_ylim(vmin_age, vmax_age)
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)
    # ax.set_xticks(ax.get_xticks()[::19])

    return ax


def plot_D_sex_individual(individual_df_sex, ax):
    stand_pal = sns.color_palette("coolwarm", n_colors=8)[::-1]

    plot_ind = sns.barplot(
        data=individual_df_sex,
        x="Co-fluctuation bin",
        y="Balanced Accuracy (%)",
        ax=ax,
        palette=stand_pal,
        ci="sd",
        errwidth=0.4,
        linewidth=0,
    )
    ax.set_ylim(vmin_sex, vmax_sex)
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)

    return ax


def plot_E_age_sex_combined(
    combined_bins_age_df, combined_bins_sex_df, full_age, full_sex, ax
):

    triu_inds = np.triu_indices(8, k=0)
    mask_age = np.zeros((8, 8), dtype=bool)
    mask_age[triu_inds] = True

    tril_inds = np.tril_indices(8, k=0)
    mask_sex = np.zeros((8, 8), dtype=bool)
    mask_sex[tril_inds] = True

    (
        age_heatmap,
        sex_heatmap,
        age_annot,
        sex_annot,
    ) = prepare_heatmap_and_annotations(
        combined_bins_age_df, combined_bins_sex_df, full_age, full_sex
    )

    sns.heatmap(
        age_heatmap.round(2),
        ax=ax,
        vmin=vmin_age,
        vmax=vmax_age,
        cmap=cmap_name_age,
        cbar=False,
        annot=True,  # age_annot,
        mask=mask_age,
        annot_kws={"fontsize": 10},
        fmt="",
    )

    plot_sex = sns.heatmap(
        sex_heatmap.round(1),
        ax=ax,
        vmin=vmin_sex,
        vmax=vmax_sex,
        cmap=cmap_name_sex,
        cbar=False,
        annot=True,  # sex_annot,
        mask=mask_sex,
        annot_kws={"fontsize": 10},
        fmt="",
    )

    plot_sex.set_xticklabels(plot_sex.get_xticklabels(), rotation=60)
    plot_sex.set_yticklabels(plot_sex.get_yticklabels(), rotation=0)

    return ax


def plot_E_colorbars(ax_age, ax_sex):

    cmap_age = plt.get_cmap(cmap_name_age)
    norm_age = matplotlib.colors.Normalize(vmin=vmin_age, vmax=vmax_age)

    cb_age = matplotlib.colorbar.ColorbarBase(
        ax_age,
        cmap=cmap_age,
        norm=norm_age,
        orientation="vertical",
        ticklocation="left",
    )
    cb_age.set_label(f"Age Prediction ({age_metric})")

    cmap_sex = plt.get_cmap(cmap_name_sex)
    norm_sex = matplotlib.colors.Normalize(vmin=vmin_sex, vmax=vmax_sex)

    cb_sex = matplotlib.colorbar.ColorbarBase(
        ax_sex,
        cmap=cmap_sex,
        norm=norm_sex,
        orientation="vertical",
        ticklocation="right",
    )
    cb_sex.set_label("Sex Classification (Balanced Accuracy %)", rotation=270)

    return ax_age, ax_sex


def suppl_figure_6():
    (
        individual_bins_df_age,
        combined_bins_df_age,
        sequential_df_age,
        full_age,
        individual_df_sex,
        combined_bins_df_sex,
        sequential_df_sex,
        full_sex,
    ) = load_transform_results()

    with plt.style.context("../style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows, grid_cols, width_ratios=width_ratios
        )

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", size=10, fontweight="bold", loc="left")
        ax1 = plot_A_age_sequential(sequential_df_age, ax1)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_title("B", size=10, fontweight="bold", loc="left")
        ax2 = plot_B_sex_sequential(sequential_df_sex, ax2)

        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("C", size=10, fontweight="bold", loc="left")
        ax3 = plot_C_age_individual(individual_bins_df_age, ax3)

        ax4 = fig.add_subplot(grid[1, 1])
        ax4.set_title("D", size=10, fontweight="bold", loc="left")
        ax4 = plot_D_sex_individual(individual_df_sex, ax4)

        ax5 = fig.add_subplot(grid[:, 3])
        ax5.set_title("E", size=10, fontweight="bold", loc="left")
        ax5 = plot_E_age_sex_combined(
            combined_bins_df_age, combined_bins_df_sex, full_age, full_sex, ax5
        )

        ax6, ax7 = fig.add_subplot(grid[:, 2]), fig.add_subplot(grid[:, 4])
        ax6, ax7 = plot_E_colorbars(ax6, ax7)

        # save images
        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    suppl_figure_6()
