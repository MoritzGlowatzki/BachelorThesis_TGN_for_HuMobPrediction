from typing import Literal

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from _1_data_IO import *
from _2a_data_preprocessing import estimate_home_location, estimate_work_location
from visualize_trajectory import (create_single_trajectory_gif, create_combined_trajectory_gif,
                                  create_single_trajectory_plot, create_combined_trajectory_plot,
                                  create_multiday_trajectory_plot)

# -------- Preliminaries -------- #
DATA_PATHS = {
    "A": "./data/original/cityA-dataset.csv",
    "B": "./data/original/cityB-dataset.csv",
    "C": "./data/original/cityC-dataset.csv",
    "D": "./data/original/cityD-dataset.csv",
}

def plot_daily_data_records_per_city():
    labels_map = ["City A", "City B", "City C", "City D"]
    colors_map = ["C0", "C1", "C2", "C3"]

    plt.figure(figsize=(15, 10))

    # -------- Plot each dataset -------- #
    for path, label, color in tqdm(zip(DATA_PATHS.values(), labels_map, colors_map), total=len(DATA_PATHS)):
        data = load_csv_file(path)
        counts = data.groupby("d").size()
        counts.plot(style=".-", label=label, color=color, markersize=12)

    # -------- Highlight weekends with gray bars and tick_labels -------- #
    for d in range(75):
        if d % 7 == 0 or d % 7 == 6:
            plt.axvspan(d - 0.5, d + 0.5, color="lightgray", alpha=1)
    plt.xticks(ticks=range(75), labels=[str(d) if d % 2 == 0 else "" for d in range(75)], fontsize=14)

    # -------- Final plot adaptations -------- #
    plt.title("Daily Data Records per City", fontsize=20, fontweight="bold")
    plt.xlabel("Day", fontsize=16)
    plt.xlim(-2, 76)
    plt.ylabel("Number of records (log scale)", fontsize=16)
    plt.yscale("log")
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()

    # -------- Show and save plot -------- #
    plt.savefig("./output/histogram_data_records_per_day.png")
    plt.show()


def plot_avg_data_records_per_hour_per_city():
    labels_map = ["City A", "City B", "City C", "City D"]
    colors_map = ["C0", "C1", "C2", "C3"]

    plt.figure(figsize=(15, 10))
    plt.axvline(x=14, color='lightgray', linewidth=3)
    plt.axvline(x=38, color='lightgray', linewidth=3)

    # -------- Plot each dataset -------- #
    for path, label, color in tqdm(zip(DATA_PATHS.values(), labels_map, colors_map), total=len(DATA_PATHS)):
        data = load_csv_file(path)
        avg_per_hour = data.groupby(["t", "d"]).size().groupby("t").mean()
        avg_per_hour.plot(style=".-", label=label, color=color, markersize=12)

    # -------- Create hourly tick_labels and vertical lines to separate day&night -------- #
    plt.xticks(ticks=range(48), labels=[f"{int(h / 2)}:00" if h % 2 == 0 else "" for h in range(48)], fontsize=14,
               rotation=45)
    plt.axvspan(-2, 14, color="lightgray", alpha=1)
    plt.axvspan(38, 50, color="lightgray", alpha=1)

    # -------- Final plot adaptations -------- #
    plt.title("Average Number of Moving Users per Half-Hour Interval per City", fontsize=20, fontweight="bold")
    plt.xlabel("Time of Day", fontsize=16)
    plt.xlim(-2, 49)
    plt.ylabel("Average number of users (log scale)", fontsize=16)
    plt.yscale("log")
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()

    # -------- Show and save plot -------- #
    plt.savefig("./output/histogram_hourly_data_records_per_day.png")
    plt.show()


def plot_gravitational_centres_for_single_user(dataset_index, uid, city_id):
    original_data = load_csv_file(DATA_PATHS[dataset_index])
    data = original_data[(original_data["uid"] == uid)][["x", "y", "t", "d"]]

    combinations = [(True, True), (True, False), (False, True)]
    titles = [
        "All Time (Home & Work Highlighted)",
        "Nighttime Only (Home Highlighted)",
        "Daytime Only (Work Highlighted)",
    ]

    # Global axis limits
    x_min, x_max = data["x"].min(), data["x"].max()
    y_min, y_max = data["y"].min(), data["y"].max()

    # Get count of least_common and most_common coordinate
    global_vmin = data[["x", "y"]].value_counts().min()
    global_vmax = data[["x", "y"]].value_counts().max()

    # Create a Normalize instance with vmin and vmax
    norm = Normalize(vmin=global_vmin, vmax=global_vmax)

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    # -------- Create subplots -------- #
    for idx, (show_home, show_work) in enumerate(combinations):
        ax = axes[idx]
        ax.set_facecolor("#440154")

        filtered_data = data.copy()
        filtered_data["weekday"] = filtered_data["d"] % 7

        if show_home and not show_work:
            # Home: early morning + evening on weekdays, entire day on weekends
            filtered_data = filtered_data[
                (
                        ((1 <= filtered_data["weekday"]) & (filtered_data["weekday"] <= 5)) &  # Weekdays: 1–5
                        (
                                ((0 <= filtered_data["t"]) & (filtered_data["t"] <= 16)) |  # 00:00–08:00
                                ((40 <= filtered_data["t"]) & (filtered_data["t"] < 48))  # 20:00–24:00
                        )
                ) |
                ((filtered_data["weekday"] == 0) | (filtered_data["weekday"] == 6))
                # Weekend: Sunday (0) & Saturday (6)
                ]

        elif not show_home and show_work:
            # Work: 9–5 only on weekdays
            filtered_data = filtered_data[
                ((1 <= filtered_data["weekday"]) & (filtered_data["weekday"] <= 5)) &  # Weekdays: 1–5
                ((18 <= filtered_data["t"]) & (filtered_data["t"] < 34))  # 09:00–17:00
                ]

        # Plot the histplot with the color normalization
        sns.histplot(data=filtered_data, x="x", y="y", cmap="viridis", thresh=0, bins=100, ax=ax, cbar=True,
                     cbar_kws={"shrink": 0.625, "label": "Absolute Density of Coordinates"},
                     vmin=global_vmin, vmax=global_vmax)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Absolute Density of Coordinates", fontsize=16)
        cbar.ax.tick_params(labelsize=12)

        # Plot the kdeplot with the same color normalization
        sns.kdeplot(data=filtered_data, x="x", y="y", fill=True, cmap="viridis", thresh=0, bw_adjust=0.5, ax=ax,
                    common_norm=norm)

        # -------- Highlight home and work location only if data is available -------- #
        if show_home:
            home_estimate = estimate_home_location(data)
            home_x, home_y = home_estimate["home_x"], home_estimate["home_y"]
            ax.scatter(home_x, home_y, color="orange", marker="x", s=75)
            ax.axvline(x=home_x, color="orange", linestyle="--", linewidth=1)
            ax.axhline(y=home_y, color="orange", linestyle="--", linewidth=1)
            ax.text(home_x + 2, home_y + 2, f"Home ({home_x}, {home_y})",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="orange", alpha=0.85, edgecolor="orange", boxstyle="round,pad=0.15"))

        if show_work:
            work_estimate = estimate_work_location(data)
            work_x, work_y = work_estimate["work_x"], work_estimate["work_y"]
            ax.scatter(work_x, work_y, color="orange", marker="x", s=75)
            ax.axvline(x=work_x, color="orange", linestyle="--", linewidth=1)
            ax.axhline(y=work_y, color="orange", linestyle="--", linewidth=1)
            ax.text(work_x + 2, work_y + 2, f"Work ({work_x}, {work_y})",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="orange", alpha=0.85, edgecolor="orange", boxstyle="round,pad=0.15"))

        # -------- Final plot adaptations -------- #
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles[idx], fontsize=18, fontweight="bold")
        ax.set_xlabel("X", fontsize=16)
        ax.set_ylabel("Y", fontsize=16)
        ax.set_aspect("equal")

    plt.suptitle(f"Gravitational Centres for uid={uid} in city {city_id}    ", fontsize=24, fontweight="bold")

    # -------- Save and show plot -------- #
    plt.savefig(f"./output/gravitational_centres_uid_{uid}.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()


def plot_gravitational_centres_all_cities():
    selected_cities = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle("Heatmaps of Log Data Counts per (x,y)-Coordinate", fontsize=24, fontweight="bold")

    for ax, city_code in zip(axes.flat, selected_cities):
        data = load_csv_file(DATA_PATHS[city_code])[["x", "y"]]

        # Plot 2D histogram: note 'x' is row (vertical), 'y' is column (horizontal)
        h = ax.hist2d(data["x"], data["y"], bins=[200, 200], cmap="viridis", norm="log")

        # Format axes
        ax.set_xlim(1, 200)
        ax.set_ylim(1, 200)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(f"City {city_code}", fontsize=16)
        ax.set_xticks([1, 50, 100, 150, 200])
        ax.set_yticks([1, 50, 100, 150, 200])
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Invert the y-axis to place (0,0) at the top-left
        ax.set_facecolor("#440154")

        # Attach a properly sized colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(h[3], cax=cax)
        cbar.set_label("Log-scaled Counts")
        if (city_code == "D"):
            cbar.set_ticks([1, 10, 100, 1000, 10000])
            cbar.set_ticklabels(["0", "10", "100", "1.000", "10.000"])
        else:
            cbar.set_ticks([1, 10, 100, 1000, 10000, 100000])
            cbar.set_ticklabels(["0", "10", "100", "1.000", "10.000", "100.000"])
        cbar.minorticks_off()

    # Save and show
    plt.savefig("./output/gravitational_centres_all_cities_original.png", dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()


def visualize_single_trajectory(dataset_index, uid, day, mode: Literal["real", "pred"], animated: bool):
    data = load_csv_file(DATA_PATHS[dataset_index])
    filtered_data = data[(data["uid"] == uid) & (data["d"] == day)]
    if animated:
        create_single_trajectory_gif(filtered_data, mode=mode)
    else:
        create_single_trajectory_plot(filtered_data, mode=mode)


def compare_real_and_predicted_trajectory(dataset_index, uid, day, animated: bool):
    data = load_csv_file(DATA_PATHS[dataset_index])
    filtered_data = data[(data["uid"] == uid) & (data["d"] == day)]
    if animated:
        create_combined_trajectory_gif(filtered_data)
    else:
        create_combined_trajectory_plot(filtered_data)


def visualize_multiday_user_trajectory(dataset_index, uid):
    data = load_csv_file(DATA_PATHS[dataset_index])
    filtered_data = data[(data["uid"] == uid)]
    create_multiday_trajectory_plot(filtered_data)


if __name__ == "__main__":
    # plot_daily_data_records_per_city()
    # plot_avg_data_records_per_hour_per_city()
    # for i in tqdm(range(0, 3), total=3):
    #     plot_gravitational_centres_for_single_user("A", i)
    plot_gravitational_centres_for_single_user("A", 0, "A")
    # plot_gravitational_centres_for_single_user("A", 14959)
    # plot_gravitational_centres_for_single_user("A", 26176)
    # plot_gravitational_centres_for_single_user("A", 60369)
    # plot_gravitational_centres_all_cities()
    # for i in tqdm(range(0, 3), total=3):
        # visualize_single_trajectory("A-small", 0, i, "real", True)
    # visualize_single_trajectory("D", 0, i, "real", False)
        # compare_real_and_predicted_trajectory("Test", 0, i, True)
        # compare_real_and_predicted_trajectory("Test", 0, i, False)
    # visualize_multiday_user_trajectory("B", 1)
    # visualize_single_trajectory("B", 1, 0, "real", False)
