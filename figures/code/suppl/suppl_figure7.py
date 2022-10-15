import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

##############################################################################
#
##############################################################################

cm = 1 / 2.54
figsize = (18 * cm, 18 * cm)
grid_rows, grid_cols = 3, 2


##############################################################################
#
##############################################################################


def load_transform_results(var="FD"):
    path = os.path.join(
        "..",
        "..",
        "..",
        "results",
        "intermediate",
        f"{var}_RSS_CORRELATIONS.csv",
    )
    results = pd.read_csv(path)
    results["correlation_method"] = results["correlation_method"].replace(
        {
            "spearman": "Spearman's r",
            "pearson": "Pearson's r",
        }
    )
    return results


def plot_dist(data_df, parcellation, correlation_method, ax):
    data = data_df.copy()
    mask = (data["correlation_method"] == correlation_method) & (
        data["parcellation"] == parcellation
    )
    data = data.loc[mask]
    data = data.rename(columns={"correlation": correlation_method})

    plot = sns.histplot(data[correlation_method], kde=True, ax=ax)
    ax.text(
        -0.3,
        200,
        f"M = {round(data[correlation_method].mean(), 2)}",
        ha="center",
        va="center",
        rotation=0,
        size=6,
        fontweight="bold",
    )
    ax.text(
        -0.3,
        180,
        f"SD = {round(data[correlation_method].std(), 2)}",
        ha="center",
        va="center",
        rotation=0,
        size=6,
        fontweight="bold",
    )
    plot.set(xlim=(-0.4, 0.4))
    plt.title(parcellation)

    return ax


def figure_histplots(var="FD", preproc="GS"):

    outfile = os.path.join(
        "..",
        "..",
        "images",
        "suppl",
        f"correlation_rss_var-{var}_preproc-{preproc}",
    )

    with plt.style.context("../style.mplstyle"):
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            grid_rows,
            grid_cols,
        )

        correlations_df = load_transform_results(var=var)
        mask_preproc = correlations_df["preproc"] == preproc
        correlations_df = correlations_df.loc[mask_preproc]

        parcellations = correlations_df["parcellation"].unique()
        corr_methods = correlations_df["correlation_method"].unique()[::-1]

        for i, parcellation in enumerate(parcellations):
            for j, corr_method in enumerate(corr_methods):
                ax = fig.add_subplot(grid[i, j])
                ax = plot_dist(correlations_df, parcellation, corr_method, ax)

        # save images
        plt.savefig(f"{outfile}.png", dpi=400)
        plt.savefig(f"{outfile}.svg", dpi=400)
        plt.savefig(f"{outfile}.pdf", dpi=400)


if __name__ == "__main__":
    vars = ["FD", "GS"]
    preprocs = ["GS", "no_GS"]
    for var, preproc in product(vars, preprocs):
        figure_histplots(var=var, preproc=preproc)
