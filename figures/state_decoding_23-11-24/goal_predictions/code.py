from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(df):
    set_plotting_style()
    ax = sns.lineplot(df, x="step", y="loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")


def plot_goal_predictions(acc_df, grid_size=5):
    set_plotting_style(font_scale=1.5)

    plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(5, 6, width_ratios=[1, 1, 1, 1, 1, 0.1])

    for j in range(5):
        for i in range(5):
            true_x, true_y = j, i

            df = acc_df[
                (acc_df["act_goal_pos_x"] == true_x)
                & (acc_df["act_goal_pos_y"] == true_y)
            ]

            preds_table = np.zeros((grid_size, grid_size))

            for row in df.itertuples():
                preds_table[row.pred_goal_pos_x, row.pred_goal_pos_y] += 1

            preds_table /= preds_table.sum()

            ax = plt.subplot(gs[j, i])
            sns.heatmap(
                preds_table,
                ax=ax,
                cbar=i == 4,
                cbar_ax=None if i != 4 else plt.subplot(gs[j, 5]),
                vmin=0,
                vmax=1,
                square=True,
            )

            if i != 0:
                ax.set_yticks([])
                ax.set_ylabel("")
            else:
                ax.set_yticks(np.array([0, 1, 2, 3, 4]) + 0.5)
                ax.set_yticklabels(["0", "1", "2", "3", "4"])

            if j != 4:
                ax.set_xticks([])
                ax.set_xlabel("")
            else:
                ax.set_xticks(np.array([0, 1, 2, 3, 4]) + 0.5)
                ax.set_xticklabels(["0", "1", "2", "3", "4"])

            width = 2
            ax.plot([true_y, true_y + 1], [true_x, true_x], "r", linewidth=width)
            ax.plot(
                [true_y, true_y + 1], [true_x + 1, true_x + 1], "r", linewidth=width
            )
            ax.plot([true_y, true_y], [true_x, true_x + 1], "r", linewidth=width)
            ax.plot(
                [true_y + 1, true_y + 1], [true_x, true_x + 1], "r", linewidth=width
            )

        cbar_ax = plt.subplot(gs[j, 5])
        cbar_ax.set_aspect(10)  # Adjust this value to fine-tune

    plt.tight_layout()


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in sorted(
            Path("figures/state_decoding_23-11-24/goal_predictions").glob("data_*.csv")
        )
    ]
    plot_goal_predictions(*data)
    plt.savefig(
        "figures/state_decoding_23-11-24/goal_predictions/goal_predictions.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
