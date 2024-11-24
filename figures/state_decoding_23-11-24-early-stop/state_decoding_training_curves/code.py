from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_state_decoding_training_curves(df):
    set_plotting_style(font_scale=2.5)

    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(df, x="step", y="loss", ax=axs[0])
    axs[0].set_xlabel("Training Step")
    axs[0].set_ylabel("Loss")
    sns.lineplot(df, x="step", y="goal_pred_acc", ax=axs[1])
    axs[1].set_xlabel("Training Step")
    axs[1].set_ylabel("Goal Prediction\nAccuracy")
    plt.tight_layout()


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in sorted(
            Path(
                "figures/state_decoding_23-11-24-early-stop/state_decoding_training_curves"
            ).glob("data_*.csv")
        )
    ]
    plot_state_decoding_training_curves(*data)
    plt.savefig(
        "figures/state_decoding_23-11-24-early-stop/state_decoding_training_curves/state_decoding_training_curves.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
