import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#############################################################################
# Constants


dataset = "hcp"
parcellation = "schaefer200x17"
unique_bins = ["1", "5", "10", "15", "20"]
model = "kernelridge_scikit"
cbpm = False
accuracy_metric, accuracy_metric_new = "test_corr", "Pearson's r"

# accuracy_metric, accuracy_metric_new = "test_r2", "r-squared"
n_targets = 25

if cbpm:
    sequential_results_path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        "cbpm",
        "predictions",
        "sequential",
        f"{dataset}_sequential_{model}_{parcellation}.csv",
    )
    individual_bins_path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        "cbpm",
        "predictions",
        "individual_bins",
        f"{dataset}_individual_bins_{model}_{parcellation}.csv",
    )
    outfile_name = os.path.join(
        "..",
        "..",
        "images",
        "suppl",
        f"{dataset}_{parcellation}_{accuracy_metric}_{model}_cbpm_sequential",
    )
else:
    sequential_results_path = os.path.join(
        "..",
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
        "..",
        "results",
        "intermediate",
        "predictions",
        "individual_bins",
        f"{dataset}_individual_bins_{model}_{parcellation}.csv",
    )

    outfile_name = os.path.join(
        "..",
        "..",
        "images",
        "suppl",
        f"{dataset}_{parcellation}_{accuracy_metric}_{model}_sequential",
    )

# figure parameters
grid_rows, grid_cols, width_ratios = 5, 6, (0.2, 0.2, 0.2, 0.2, 0.2, 0.02)
vmin, vmax = 0, 0.6
cm = 1 / 2.54

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
        markers=True,
        markeredgewidth=0.3,
        markeredgecolor="k",
        dashes=False
        # linewidth=10,
        # markersize=20
    )
    ax.set_ylim(vmin, vmax)

    plt.title(target, fontsize=5)

    if grid_col != 2 or grid_row != 0:
        try:
            ax.get_legend().remove()
        except AttributeError:
            print("Ok")
    else:
        leg = ax.legend(loc="upper center", ncol=2, fancybox=True, fontsize=5)
        for line in leg.get_lines():
            line.set_linewidth(1.5)
            line.set_markersize(3)

    if grid_col != 0:
        ax.set_yticks([])
        ax.set_ylabel("")

    if grid_row != 4:
        ax.set_xticks([])
        ax.set_xlabel("")

    return ax


def suppl_figure_2():
    sequential_df, targets = load_transform_results()

    with plt.style.context("../style.mplstyle"):
        fig = plt.figure(figsize=(18 * cm, 18 * cm))
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
            if grid_col != 4:
                grid_col += 1
            else:
                grid_col = 0
                grid_row += 1

        plt.savefig(f"{outfile_name}.png", dpi=400)
        plt.savefig(f"{outfile_name}.svg", dpi=400)
        plt.savefig(f"{outfile_name}.pdf", dpi=400)


if __name__ == "__main__":
    suppl_figure_2()
