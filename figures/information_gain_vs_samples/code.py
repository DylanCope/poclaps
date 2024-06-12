from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_information_gain(df: pd.DataFrame, max_gain: float):
    set_plotting_style(font_scale=1.5)
    ax = sns.lineplot(data=df, x="k", y="gain", errorbar="sd")
    ax.axhline(max_gain, color="black", linestyle="--", label="Max Gain", alpha=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Information Gain (bits)")
    ax.legend(loc="lower right")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/information_gain_vs_samples").glob("data_*.csv")
    ]
    plot_information_gain(*data, max_gain=4.643856189774724)
    plt.savefig(
        "figures/information_gain_vs_samples/information_gain_vs_samples.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
