from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


policy_samples = 25


def plot_true_goal_rate(stochastic_policy_df):
    set_plotting_style(font_scale=2)
    plt.figure(figsize=(10, 6))
    max_policy_samples = stochastic_policy_df.policy_samples.max()
    ax = sns.lineplot(
        data=stochastic_policy_df[
            stochastic_policy_df.policy_samples == max_policy_samples
        ],
        x="k",
        y="true_goal_in_possible",
    )
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("True Goal In Possible Goals")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/true_goal_rate_vs_samples").glob("data_*.csv")
    ]
    plot_true_goal_rate(*data)
    plt.savefig(
        "figures/true_goal_rate_vs_samples/true_goal_rate_vs_samples.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
