from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

##############################################################################
# Constants


results_dir = f"../../results"
n_bins = 20

cm = 1 / 2.54
figsize = (18 * cm, 9 * cm)
grid_rows, grid_cols = 1, 2

vmin_heatmap, vmax_heatmap = 0, None
cmap = "autumn"


###############################################################################
# load data for combined and individual bins and put in heatmap
###############################################################################


path_individual_bins = f"{results_dir}/SC_FC_corr_individual_bins/hcp/GS"

fname_individual_bins = (
    "dataset-hcp_preprocessing_level-GS_parcellation-schaefer200x17"
    "_criterion-rss_corrmethod-pearson.tsv"
)

# combined bins
path_combined_bins = f"{results_dir}/SC_FC_corr_combined_bins/hcp/GS"

fname_combined_bins = (
    "dataset-hcp_preprocessing_level-GS_parcellation-schaefer200x17"
    "_criterion-rss_corrmethod-pearson.tsv"
)
fname_sequential = (
    "dataset-hcp_preprocessing_level-GS_parcellation-schaefer200x17"
    "_criterion-rss_corrmethod-pearson.tsv"
)
path = f"{results_dir}/SC_FC_corr_sequential/hcp/GS"
path_to_sequential = f"{path}/{fname_sequential}"

outfile = f"../images/paper/figure8_FC_SC_Correlations"


def load_transform_results():
    sc_fc_individual_bins = pd.read_csv(
        f"{path_individual_bins}/{fname_individual_bins}", sep="\t"
    ).drop(columns="full")
    sc_fc_individual_bins_avg = sc_fc_individual_bins.mean(axis=0)
    sc_fc_combined_bins = pd.read_csv(
        f"{path_combined_bins}/{fname_combined_bins}", sep="\t"
    )
    sc_fc_combined_bins_avg = sc_fc_combined_bins.mean()
    sc_fc_sequential = pd.read_csv(path_to_sequential, sep="\t")
    sc_fc_sequential, min_max = format_sequential(sc_fc_sequential)
    sc_fc_sequential["Co-fluctuation level"] = sc_fc_sequential[
        "Co-fluctuation level"
    ].replace({"high": "HACF", "low": "LACF"})

    return (
        sc_fc_individual_bins_avg,
        sc_fc_combined_bins_avg,
        sc_fc_sequential,
        min_max,
    )


def format_sequential(sc_fc_sequential):
    thresholds = [1, 2, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    ranks = ["high", "low"]
    plot_dict = {
        "Pearson's r (FC~SC)": [],
        "Co-fluctuation level": [],
        "Threshold (%)": [],
    }
    series_high_min = []
    series_high_max = []
    series_low_min = []
    series_low_max = []
    threshes = []

    for threshold, rank in product(thresholds, ranks):
        col_name = f"{threshold}_{rank}"

        plot_dict["Pearson's r (FC~SC)"].append(
            sc_fc_sequential[col_name].mean()
        )
        plot_dict["Co-fluctuation level"].append(rank)
        plot_dict["Threshold (%)"].append(threshold)

        if rank in ["high"]:
            series_high_min.append(sc_fc_sequential[col_name].min())
            series_high_max.append(sc_fc_sequential[col_name].max())
            threshes.append(threshold)
        else:
            series_low_min.append(sc_fc_sequential[col_name].min())
            series_low_max.append(sc_fc_sequential[col_name].max())

    return pd.DataFrame(plot_dict), pd.DataFrame(
        {
            "high_max": series_high_max,
            "high_min": series_high_min,
            "low_max": series_low_max,
            "low_min": series_low_min,
            "thresholds": threshes,
        }
    )


def prepare_heatmap(sc_fc_combined_bins_avg, sc_fc_individual_bins_avg):
    heatmap = np.zeros((n_bins, n_bins))
    for i, j in product(range(1, n_bins + 1), range(1, n_bins + 1)):
        if f"{i}_{j}" in sc_fc_combined_bins_avg.index:
            heatmap[j - 1, i - 1] = sc_fc_combined_bins_avg.loc[f"{i}_{j}"]

    np.fill_diagonal(heatmap, sc_fc_individual_bins_avg.values)
    heatmap_df = pd.DataFrame(heatmap)
    heatmap_df.index = [f"Bin {x}" for x in range(1, 21)]
    heatmap_df.columns = [f"Bin {x}" for x in range(1, 21)]

    return heatmap_df


def plot_A_heatmap(sc_fc_individual_bins_avg, sc_fc_combined_bins_avg, ax):

    heatmap_df = prepare_heatmap(
        sc_fc_combined_bins_avg, sc_fc_individual_bins_avg
    )

    triu_inds = np.triu_indices(n_bins, k=1)
    mask_heatmap = np.zeros((n_bins, n_bins), dtype=bool)
    mask_heatmap[triu_inds] = True

    plot = sns.heatmap(
        heatmap_df,
        ax=ax,
        vmin=vmin_heatmap,
        vmax=vmax_heatmap,
        cmap=cmap,
        cbar=True,
        annot=False,
        mask=mask_heatmap,
        annot_kws={"size": 10},
    )
    # plot.set_yticklabels(plot.get_yticks(), size = 15)
    # plot.set_xticklabels(plot.get_xticks(), size = 15)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)

    return ax


def plot_B_sequential(sc_fc_sequential, seq_min_max, ax):

    sns.lineplot(
        ax=ax,
        data=sc_fc_sequential,
        hue="Co-fluctuation level",
        style="Co-fluctuation level",
        y="Pearson's r (FC~SC)",
        x="Threshold (%)",
        palette=["r", "b"],
        markers=True,
        ci=None,
        dashes=False,
        markeredgecolor="k",
        markeredgewidth=0.3,
    )

    for line in ax.legend(
        bbox_to_anchor=(0.9, 1), ncol=1, fontsize=6
    ).get_lines():
        line.set_linewidth(1.5)
        line.set_markersize(3)

    ax.fill_between(
        seq_min_max["thresholds"],
        y1=seq_min_max["low_min"],
        y2=seq_min_max["low_max"],
        alpha=0.3,
        color="blue",
    )

    ax.fill_between(
        seq_min_max["thresholds"],
        y1=seq_min_max["high_min"],
        y2=seq_min_max["high_max"],
        alpha=0.3,
        color="red",
    )

    return ax


def figure_eight():
    (
        sc_fc_individual_bins_avg,
        sc_fc_combined_bins_avg,
        sc_fc_sequential,
        seq_min_max,
    ) = load_transform_results()

    with plt.style.context("./style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,
        )

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title("A", fontweight="bold", loc="left", fontsize=10)
        ax1 = plot_A_heatmap(
            sc_fc_individual_bins_avg, sc_fc_combined_bins_avg, ax1
        )

        ax4 = fig.add_subplot(grid[0, 1])
        ax4.set_title("B", fontweight="bold", loc="left", fontsize=10)
        ax4 = plot_B_sequential(sc_fc_sequential, seq_min_max, ax4)

        # save images

        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    figure_eight()
