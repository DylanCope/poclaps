from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


policy_samples = 25


def plot_information_gain(
    df: pd.DataFrame, wrong_policy_df, stochastic_policy_df, max_gain: float
):
    set_plotting_style(font_scale=2)
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x="k",
        y="gain",
        errorbar=lambda x: (x.min(), x.max()),
        label="Correct Optimal Policy",
    )
    sns.lineplot(
        data=wrong_policy_df,
        x="k",
        y="gain",
        ax=ax,
        errorbar=lambda x: (x.min(), x.max()),
        label="Incorrect Optimal Policy",
    )
    max_policy_samples = stochastic_policy_df.policy_samples.max()
    sns.lineplot(
        data=stochastic_policy_df[
            stochastic_policy_df.policy_samples == max_policy_samples
        ],
        x="k",
        y="gain",
        ax=ax,
        errorbar=lambda x: (x.min(), x.max()),
        label=f"{max_policy_samples+1} Randomly Sampled\nOptimal Policies",
    )
    ax.axhline(max_gain, color="black", linestyle="--", label="Max Gain", alpha=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Information Gain (bits)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.7, 1))


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/information_gain_vs_samples_diff_policies").glob(
            "data_*.csv"
        )
    ]
    plot_information_gain(*data, max_gain=4.643856189774724)
    plt.savefig(
        "figures/information_gain_vs_samples_diff_policies/information_gain_vs_samples_diff_policies.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
