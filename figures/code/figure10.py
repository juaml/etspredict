#!/usr/bin/env python3


import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from baycomp import two_on_single

from etspredict.prediction.scoring_utils import compare_propabilites

# -------------------------------------------------------------------------- #
# Prepre input paths


dataset = "hcp"
atlas = "schaefer200x17"
sex_pred_model = "ridgeclassifier"

cmap_name_age = "autumn"
cmap_name_sex = "autumn"

rope_perc = 0.05
rope_perc_sex = 5

outfile = f"../images/paper/sc_free_age_sex"

# figure parameters
cmap_name_age = "autumn"
cmap_name_sex = "autumn"

vmin_sex, vmax_sex = 50, 75
vmin_age, vmax_age = 0, 0.6

grid_rows, grid_cols = 2, 5
width_ratios = (0.20, 0.20, 0.02, 0.56, 0.02)

cm = 1 / 2.54
figsize = (18 * cm, 9 * cm)

# age predictions
path_age_prediction = os.path.join(
    "..", "..", "results", "intermediate", "age_predictions_sc_free"
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
    "..", "..", "results", "intermediate", "sex_predictions_sc_free"
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


def load_transform_results():
    bin_dict = {}
    for i in range(1, 21):
        bin_dict[str(i)] = f"Bin {i}"

    individual_bins_df_age = pd.read_csv(individual_bins_path_age)
    individual_bins_df_age = individual_bins_df_age.rename(
        columns={
            "ampl_rankings": "Co-fluctuation bin",
            "threshold": "Threshold (%)",
            "test_corr": "Pearson's r",
        }
    )
    mask_full_age = individual_bins_df_age["Co-fluctuation bin"] == "full"
    full_age = individual_bins_df_age.loc[mask_full_age]

    ind_df_mask = individual_bins_df_age["Co-fluctuation bin"] != "full"
    individual_bins_df_age = individual_bins_df_age.loc[ind_df_mask]
    individual_bins_df_age["Co-fluctuation bin"] = individual_bins_df_age[
        "Co-fluctuation bin"
    ].replace(bin_dict)

    combined_bins_df_age = pd.read_csv(combined_bins_path_age)
    sequential_df_age = pd.read_csv(sequential_path_age)

    sequential_df_age = sequential_df_age.rename(
        {
            "test_corr": "Pearson's r",
            "threshold": "Threshold (in %)",
            "ampl_rankings": "Co-fluctuation level",
        },
        axis="columns",
    )
    sequential_df_age["Co-fluctuation level"] = sequential_df_age[
        "Co-fluctuation level"
    ].replace("high", "HACF")

    sequential_df_age["Co-fluctuation level"] = sequential_df_age[
        "Co-fluctuation level"
    ].replace("low", "LACF")

    sequential_df_sex = pd.read_csv(sequential_path_sex)
    sequential_df_sex = sequential_df_sex.rename(
        {
            "test_accuracy": "Accuracy (%)",
            "threshold": "Threshold (in %)",
            "ampl_rankings": "Co-fluctuation level",
        },
        axis="columns",
    )
    sequential_df_sex["Co-fluctuation level"] = sequential_df_sex[
        "Co-fluctuation level"
    ].replace("high", "HACF")

    sequential_df_sex["Co-fluctuation level"] = sequential_df_sex[
        "Co-fluctuation level"
    ].replace("low", "LACF")
    sequential_df_sex["Accuracy (%)"] = sequential_df_sex["Accuracy (%)"] * 100

    individual_df_sex = pd.read_csv(individual_path_sex)
    individual_df_sex = individual_df_sex.rename(
        {
            "test_accuracy": "Accuracy (%)",
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation bin",
        },
        axis="columns",
    )

    individual_df_sex["Accuracy (%)"] = individual_df_sex["Accuracy (%)"] * 100
    mask_full_sex = individual_df_sex["Co-fluctuation bin"] == "full"
    full_sex = individual_df_sex.loc[mask_full_sex]

    ind_df_mask = individual_df_sex["Co-fluctuation bin"] != "full"
    individual_df_sex = individual_df_sex.loc[ind_df_mask]
    individual_df_sex["Co-fluctuation bin"] = individual_df_sex[
        "Co-fluctuation bin"
    ].replace(bin_dict)
    combined_bins_df_sex = pd.read_csv(combined_bins_path_sex)

    combined_bins_df_sex["test_accuracy"] = (
        combined_bins_df_sex["test_accuracy"] * 100
    )

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
    age_heatmap = np.zeros((20, 20))
    sex_heatmap = np.zeros((20, 20))

    age_annot = np.empty((20, 20), dtype="<U4")
    sex_annot = np.empty((20, 20), dtype="<U4")

    ampl_rankings_age = set(combined_bins_age_df["ampl_rankings"])
    ampl_rankings_sex = set(combined_bins_sex_df["ampl_rankings"])

    for rank in ampl_rankings_age:
        age_mask = combined_bins_age_df["ampl_rankings"] == rank
        rank_df = combined_bins_age_df.loc[age_mask]
        avg_score_age_rank = rank_df["test_corr"].mean()
        idx1, idx2 = rank.split("_")
        age_heatmap[int(idx2) - 1, int(idx1) - 1] = avg_score_age_rank

        # first do rope analysis
        # embed()
        p_left, p_rope, p_right = two_on_single(
            rank_df["test_corr"].values,
            full_age["Pearson's r"].values,
            rope_perc,
        )
        comparison = compare_propabilites(p_left, p_rope, p_right)
        age_annot[int(idx2) - 1, int(idx1) - 1] = comparison

    for rank in ampl_rankings_sex:
        sex_mask = combined_bins_sex_df["ampl_rankings"] == rank
        rank_df = combined_bins_sex_df.loc[sex_mask]
        avg_score_sex_rank = rank_df["test_accuracy"].mean()
        idx1, idx2 = rank.split("_")
        sex_heatmap[int(idx1) - 1, int(idx2) - 1] = avg_score_sex_rank

        # first do rope analysis

        p_left, p_rope, p_right = two_on_single(
            rank_df["test_accuracy"].values,
            full_sex["Accuracy (%)"].values,
            rope_perc_sex,
        )

        comparison = compare_propabilites(p_left, p_rope, p_right)
        sex_annot[int(idx1) - 1, int(idx2) - 1] = comparison

    bins = [f"Bin {x}" for x in range(1, 21)]
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
        y="Pearson's r",
        x="Threshold (in %)",
        palette=["r", "b"],
        markers=True,
        ci="sd",
        markeredgewidth=0.3,
        dashes=False,
        markeredgecolor="k",
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
        y="Accuracy (%)",
        x="Threshold (in %)",
        palette=["r", "b"],
        markers=True,
        ci="sd",
        markeredgewidth=0.3,
        dashes=False,
        markeredgecolor="k",
    )

    for line in ax.legend(
        bbox_to_anchor=(
            0.5,
            0.5,
        ),
        fontsize=6,
    ).get_lines():
        line.set_linewidth(1.5)
        line.set_markersize(3)

    ax.set_ylim(vmin_sex, vmax_sex)

    return ax


def plot_C_age_individual(individual_bins_df_age, ax):

    stand_pal = sns.color_palette("coolwarm", n_colors=20)[::-1]

    plot_ind = sns.barplot(
        data=individual_bins_df_age,
        x="Co-fluctuation bin",
        y="Pearson's r",
        ax=ax,
        palette=stand_pal,
        ci="sd",
        errwidth=0.4,
        linewidth=0,
    )
    ax.set_ylim(vmin_age, vmax_age)
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(ax.get_xticks()[::19])

    return ax


def plot_D_sex_individual(individual_df_sex, ax):
    stand_pal = sns.color_palette("coolwarm", n_colors=20)[::-1]

    plot_ind = sns.barplot(
        data=individual_df_sex,
        x="Co-fluctuation bin",
        y="Accuracy (%)",
        ax=ax,
        ci="sd",
        palette=stand_pal,
        errwidth=0.4,
        linewidth=0,
    )
    ax.set_ylim(vmin_sex, vmax_sex)
    plot_ind.set_xticklabels(plot_ind.get_xticklabels(), rotation=60)
    ax.set_xticks(ax.get_xticks()[::19])

    return ax


def plot_E_age_sex_combined(
    combined_bins_age_df, combined_bins_sex_df, full_age, full_sex, ax
):

    triu_inds = np.triu_indices(20, k=0)
    mask_age = np.zeros((20, 20), dtype=bool)
    mask_age[triu_inds] = True

    tril_inds = np.tril_indices(20, k=0)
    mask_sex = np.zeros((20, 20), dtype=bool)
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
        age_heatmap,
        ax=ax,
        vmin=vmin_age,
        vmax=vmax_age,
        cmap=cmap_name_age,
        cbar=False,
        annot=age_annot,
        annot_kws={"fontsize": 10},
        mask=mask_age,
        fmt="",
    )

    plot_sex = sns.heatmap(
        sex_heatmap,
        ax=ax,
        vmin=vmin_sex,
        vmax=vmax_sex,
        cmap=cmap_name_sex,
        cbar=False,
        annot=sex_annot,
        annot_kws={"fontsize": 10},
        mask=mask_sex,
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
    cb_age.set_label(
        "Age Prediction (Pearson's r)",
    )

    cmap_sex = plt.get_cmap(cmap_name_sex)
    norm_sex = matplotlib.colors.Normalize(vmin=vmin_sex, vmax=vmax_sex)

    cb_sex = matplotlib.colorbar.ColorbarBase(
        ax_sex,
        cmap=cmap_sex,
        norm=norm_sex,
        orientation="vertical",
        ticklocation="right",
    )
    cb_sex.set_label("Sex Classification (Accuracy %)", rotation=270)

    return ax_age, ax_sex


def figure_ten():
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

    with plt.style.context("style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows, grid_cols, width_ratios=width_ratios
        )

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("C", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_A_age_sequential(sequential_df_age, ax1)

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_title("D", fontweight="bold", loc="left", fontsize=10)
        ax2 = plot_B_sex_sequential(sequential_df_sex, ax2)

        ax3 = fig.add_subplot(grid[1, 0])
        ax3.set_title("E", fontweight="bold", loc="left", fontsize=10)
        ax3 = plot_C_age_individual(individual_bins_df_age, ax3)

        ax4 = fig.add_subplot(grid[1, 1])
        ax4.set_title("F", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_D_sex_individual(individual_df_sex, ax4)

        ax5 = fig.add_subplot(grid[:, 3])
        ax5.set_title("G", fontweight="bold", loc="left", fontsize=10)
        ax5 = plot_E_age_sex_combined(
            combined_bins_df_age, combined_bins_df_sex, full_age, full_sex, ax5
        )

        ax6, ax7 = fig.add_subplot(grid[:, 2]), fig.add_subplot(grid[:, 4])
        ax6, ax7 = plot_E_colorbars(ax6, ax7)

        # save images
        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)


if __name__ == "__main__":
    figure_ten()
