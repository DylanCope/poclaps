from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("pdf")


def plot_policy_training_history(policy_training_metrics):
    set_plotting_style(font_scale=1.5)
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        policy_training_metrics["iteration"], policy_training_metrics["mean_reward"]
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/prelim_run_results/policy_training_history").glob(
            "data_*.csv"
        )
    ]
    plot_policy_training_history(*data)
    plt.savefig(
        "figures/prelim_run_results/policy_training_history/policy_training_history.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
