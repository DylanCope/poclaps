from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(df):
    set_plotting_style()
    ax = sns.lineplot(df, x="step", y="loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in sorted(Path("figures/transition_model_loss").glob("data_*.csv"))
    ]
    plot(*data)
    plt.savefig(
        "figures/transition_model_loss/transition_model_loss.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
