from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


T = 4


def plot_policy_location_heatmap(agent_pos_df, grid_size, n_samples):
    agent_pos_df["idx"] = agent_pos_df["X"] * grid_size + agent_pos_df["Y"]
    location_heatmap = (
        agent_pos_df.groupby(["X", "Y"])["idx"]
        .count()
        .reset_index()
        .pivot(index="X", columns="Y", values="idx")
    )
    location_heatmap[grid_size - 1][grid_size - 1] = n_samples
    location_heatmap[0][0] = n_samples

    set_plotting_style(font_scale=2)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        location_heatmap.T / n_samples,
        cmap="viridis",
        cbar=True,
        annot=True,
        fmt=".3f",
        square=True,
    )


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/prelim_run_results/agent_pos_heatmap").glob(
            "data_*.csv"
        )
    ]
    plot_policy_location_heatmap(*data, grid_size=5, n_samples=512)
    plt.savefig(
        "figures/prelim_run_results/agent_pos_heatmap/agent_pos_heatmap.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
