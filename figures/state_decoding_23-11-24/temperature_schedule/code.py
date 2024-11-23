from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_temp_schedules(schedule_df: pd.DataFrame):
    set_plotting_style(font_scale=2.5)
    _, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x="step", y="temp", data=schedule_df, ax=ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in sorted(
            Path("figures/state_decoding_23-11-24/temperature_schedule").glob(
                "data_*.csv"
            )
        )
    ]
    plot_temp_schedules(*data)
    plt.savefig(
        "figures/state_decoding_23-11-24/temperature_schedule/temperature_schedule.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
