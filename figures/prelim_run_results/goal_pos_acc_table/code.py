from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_acc_table(msg_acc_df, grid_size=5):
    msg_acc_df["Goal X"] = msg_acc_df["Message"] // grid_size
    msg_acc_df["Goal Y"] = msg_acc_df["Message"] % grid_size

    acc_table = (
        msg_acc_df.groupby(["Goal X", "Goal Y"])["Accuracy"]
        .mean()
        .reset_index()
        .pivot(index="Goal X", columns="Goal Y", values="Accuracy")
    )

    set_plotting_style(font_scale=2)

    ax = sns.heatmap(
        np.float32(acc_table.to_numpy()),
        cmap="viridis",
        cbar=True,
        annot=True,
        fmt=".2f",
    )
    ax.set_xlabel("Goal X")
    ax.set_ylabel("Goal Y")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/prelim_run_results/goal_pos_acc_table").glob("data_*.csv")
    ]
    plot_acc_table(*data)
    plt.savefig(
        "figures/prelim_run_results/goal_pos_acc_table/goal_pos_acc_table.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
