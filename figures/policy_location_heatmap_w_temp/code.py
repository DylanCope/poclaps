from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_policy_location_heatmap(agent_pos_df, grid_size, n_samples):
    agent_pos_df["idx"] = agent_pos_df["X"] * grid_size + agent_pos_df["Y"]

    set_plotting_style(font_scale=3)

    _, axs = plt.subplots(1, 3, figsize=(30, 8))
    for i, temp in enumerate([1.0, 0.1, 0.01]):
        df = agent_pos_df[agent_pos_df["temp"] == temp]
        location_heatmap = (
            df.groupby(["X", "Y"])["idx"]
            .count()
            .reset_index()
            .pivot(index="X", columns="Y", values="idx")
        )
        location_heatmap[grid_size - 1][grid_size - 1] = n_samples
        location_heatmap[0][0] = n_samples
        ax = axs[i]
        sns.heatmap(
            location_heatmap.T / n_samples,
            cmap="viridis",
            cbar=True,
            annot=True,
            fmt=".2f",
            square=True,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Temperature: {temp}", pad=20)


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in sorted(
            Path("figures/policy_location_heatmap_w_temp").glob("data_*.csv")
        )
    ]
    plot_policy_location_heatmap(*data, grid_size=5, n_samples=512)
    plt.savefig(
        "figures/policy_location_heatmap_w_temp/policy_location_heatmap_w_temp.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
