import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from baycomp import two_on_single

from etspredict.prediction.scoring_utils import compare_propabilites

#############################################################################
# Constants


dataset = "hcp"
parcellation = "schaefer200x17"
unique_bins = ["1", "5", "10", "15", "20"]
model = "kernelridge_scikit"
accuracy_metric, accuracy_metric_new = "test_corr", "Pearson's r"
n_targets = 9

sequential_results_path = os.path.join(
    "..",
    "..",
    "results",
    "intermediate",
    "predictions",
    "sequential",
    f"{dataset}_sequential_{model}_{parcellation}.csv",
)
individual_bins_path = os.path.join(
    "..",
    "..",
    "results",
    "intermediate",
    "predictions",
    "individual_bins",
    f"{dataset}_individual_bins_{model}_{parcellation}.csv",
)

outfile_name = os.path.join(
    "..",
    "images",
    "paper",
    f"Figure3_{dataset}_{parcellation}_{accuracy_metric}_sequential",
)

# figure parameters
grid_rows, grid_cols, width_ratios = 3, 4, (0.32, 0.32, 0.32, 0.02)
vmin, vmax = 0, 0.5

cm = 1 / 2.54
figsize = (18 * cm, 14 * cm)

#############################################################################
#


def load_transform_results():
    sequential_df = pd.read_csv(sequential_results_path)

    individual_bins_df = pd.read_csv(individual_bins_path)
    full_fc_mask = individual_bins_df["ampl_rankings"] == "full"
    full_fc_df = individual_bins_df.loc[full_fc_mask]

    targets = select_targets(full_fc_df)

    sequential_df["ampl_rankings"] = sequential_df["ampl_rankings"].replace(
        {"high": "HACF", "low": "LACF"}
    )

    sequential_df = sequential_df.rename(
        columns={
            accuracy_metric: accuracy_metric_new,
            "threshold": "Threshold (%)",
            "ampl_rankings": "Co-fluctuation level",
        }
    )

    return sequential_df, targets


def select_targets(full_fc_df):
    full_fc_ordered_by_score = (
        full_fc_df.groupby("target")
        .mean()
        .sort_values(by=accuracy_metric, ascending=False)
    )

    return list(full_fc_ordered_by_score.index)[:n_targets]


def plot_target(target_df, ax, grid_row, grid_col, target):

    sns.lineplot(
        ax=ax,
        data=target_df,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y=accuracy_metric_new,
        ci="sd",
        x="Threshold (%)",
        palette=["r", "b"],
        markers=True,  # ["o", "^"],
        linewidth=1.5,
        markersize=3,
        markeredgecolor="k",
        markeredgewidth=0.3,
        dashes=False,
        # err_style="bars",
        # err_kws=err_kws,
    )
    ax.set_ylim(vmin, vmax)

    if target in ["Reading (pronounciation)"]:
        target_label = "Reading (pronunciation)"
    else:
        target_label = target

    plt.title(target_label)

    if grid_col != 2 or grid_row != 1:
        try:
            ax.get_legend().remove()
        except AttributeError:
            print("Ok")
    else:
        leg = ax.legend(
            bbox_to_anchor=(1, 1), ncol=1, fancybox=True, fontsize=8
        )
        for line in leg.get_lines():
            line.set_linewidth(1)
            line.set_markersize(1)

    if grid_col != 0:
        ax.set_yticks([])
        ax.set_ylabel("")

    if grid_row != 2:
        ax.set_xticks([])
        ax.set_xlabel("")

    for thresh in target_df["Threshold (%)"].unique():
        co_data_mask = target_df["Threshold (%)"] == thresh
        co_data = target_df.loc[co_data_mask]

        hacf_mask = co_data["Co-fluctuation level"] == "HACF"
        lacf_mask = co_data["Co-fluctuation level"] == "LACF"

        hacf = co_data.loc[hacf_mask]
        lacf = co_data.loc[lacf_mask]
        p_left, p_rope, p_right = two_on_single(
            lacf[accuracy_metric_new].values,
            hacf[accuracy_metric_new].values,
            0.05,
        )
        compare_propabilites(p_left, p_rope, p_right)

    return ax


def figure_three():
    sequential_df, targets = load_transform_results()

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows, grid_cols, width_ratios=width_ratios
        )

        grid_row, grid_col = 0, 0

        for target in targets:

            target_mask = sequential_df["target"] == target
            target_df = sequential_df.loc[target_mask]

            ax1 = fig.add_subplot(grid[grid_row, grid_col])
            ax1 = plot_target(target_df, ax1, grid_row, grid_col, target)

            # update grid indices
            if grid_col != 2:
                grid_col += 1
            else:
                grid_col = 0
                grid_row += 1

        sns.despine(fig, top=True, right=True)
        # plt.tight_layout()

        plt.savefig(f"{outfile_name}.png", dpi=400)
        plt.savefig(f"{outfile_name}.svg", dpi=400)
        plt.savefig(f"{outfile_name}.pdf", dpi=400)


if __name__ == "__main__":
    figure_three()
