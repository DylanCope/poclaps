from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("pdf")


def plot_data(data):
    set_plotting_style(rc={"font.family": "DejaVu Sans"})
    plt.scatter(data.x, data.y, s=1, alpha=0.5, c="#9b59b6")
    plt.axis("equal")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.title("Data $x \sim p(x)$")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/sampled_data").glob("data_*.csv")
    ]
    plot_data(*data)
    plt.savefig("figures/sampled_data/sampled_data.pdf", bbox_inches="tight", dpi=1000)


if __name__ == "__main__":
    reproduce_figure()
