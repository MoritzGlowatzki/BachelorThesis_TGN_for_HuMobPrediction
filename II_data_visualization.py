import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

from III_data_preprocessing import estimate_home_location, estimate_work_location
from I_data_IO import *
from trajectory_visualization import create_single_trajectory_gif

# -------- Preliminaries -------- #
DATA_PATHS = {
    "A": "./data/cityA-dataset.csv",
    "B": "./data/cityB-dataset.csv",
    "C": "./data/cityC-dataset.csv",
    "D": "./data/cityD-dataset.csv",
    "A-small": "./data/cityA-dataset-small.csv"
}
RAW_SMALL_DATA_PATH = "./data/cityA-dataset-small.csv"


def plot_daily_data_records_per_city():
    labels_map = ["City A", "City B", "City C", "City D"]
    colors_map = ["C0", "C1", "C2", "C3"]

    plt.figure(figsize=(15, 5))

    # -------- Highlight weekends with gray bars and tick_labels -------- #
    x_labels = []
    for d in range(75):
        if d % 7 == 0 or d % 7 == 6:
            plt.axvspan(d - 0.5, d + 0.5, color="lightgray", alpha=1)
            x_labels.append(str(d))
        else:
            x_labels.append("")
    plt.xticks(ticks=range(75), labels=x_labels, fontsize=8, rotation=0)

    # -------- Plot each dataset -------- #
    for path, label, color in tqdm(zip(DATA_PATHS, labels_map, colors_map), total=len(DATA_PATHS)):
        data = load_csv_file(path)
        counts = data.groupby("d").size()
        counts.plot(style=".-", label=label, color=color)

    # -------- Final plot adaptations -------- #
    plt.title("Daily Data Records per City")
    plt.xlabel("Day")
    plt.ylabel("Number of records (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    # -------- Show and save plot -------- #
    plt.savefig("./output/histogram_data_records_per_day.png")
    plt.show()


def plot_gravitational_centres(dataset_index, uid):
    original_data = load_csv_file(DATA_PATHS[dataset_index])
    data = original_data[(original_data["uid"] == uid)][["x", "y", "t"]]

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
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    # -------- Create subplots -------- #
    for idx, (show_home, show_work) in enumerate(combinations):
        ax = axes[idx]
        ax.set_facecolor("#440154")

        filtered_data = data.copy()
        if show_home and not show_work:
            filtered_data = filtered_data[((0 <= filtered_data["t"]) & (filtered_data["t"] <= 12))
                                          | ((44 <= filtered_data["t"]) & (filtered_data["t"] < 48))]
        elif not show_home and show_work:
            filtered_data = filtered_data[(18 <= filtered_data["t"]) & (filtered_data["t"] < 34)]

        # Plot the histplot with the color normalization
        sns.histplot(data=filtered_data, x="x", y="y", cmap="viridis", thresh=0, bins=100, ax=ax, cbar=True,
                     cbar_kws={"shrink": 0.625, "label": "Absolute Density of Coordinates"},
                     vmin=global_vmin, vmax=global_vmax)

        # Plot the kdeplot with the same color normalization
        sns.kdeplot(data=filtered_data, x="x", y="y", fill=True, cmap="viridis", thresh=0, bw_adjust=0.5, ax=ax,
                    common_norm=norm)

        # -------- Highlight home and work location -------- #
        if show_home:
            home_x, home_y = estimate_home_location(filtered_data)
            ax.scatter(home_x, home_y, color="orange", marker="x", s=75)
            ax.axvline(x=home_x, color="orange", linestyle="--", linewidth=1)
            ax.axhline(y=home_y, color="orange", linestyle="--", linewidth=1)
            ax.text(home_x + 2, home_y + 2, f"Home ({home_x}, {home_y})",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor='orange', alpha=0.85, edgecolor='orange', boxstyle='round,pad=0.15'))

        if show_work:
            work_x, work_y = estimate_work_location(filtered_data)
            ax.scatter(work_x, work_y, color="orange", marker="x", s=75)
            ax.axvline(x=work_x, color="orange", linestyle="--", linewidth=1)
            ax.axhline(y=work_y, color="orange", linestyle="--", linewidth=1)
            ax.text(work_x + 2, work_y + 2, f"Work ({work_x}, {work_y})",
                    color="white", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor='orange', alpha=0.85, edgecolor='orange', boxstyle='round,pad=0.15'))

        # -------- Final plot adaptations -------- #
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles[idx], fontsize=16, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    plt.suptitle(f"Gravitational Centres for uid={uid}", fontsize=24, fontweight='bold')

    # -------- Save and show plot -------- #
    plt.savefig(f"./output/gravitational_centres_uid_{uid}.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()


def visualize_trajectory(dataset_index, uid, day):
    data = load_csv_file(DATA_PATHS[dataset_index])
    filtered_data = data[(data["uid"] == uid) & (data["d"] == day)]
    create_single_trajectory_gif(filtered_data)


if __name__ == "__main__":
    # plot_daily_data_records_per_city()
    for i in tqdm(range(0, 3), total=3):
        plot_gravitational_centres("A", i)
    # plot_gravitational_centres("A-small", 0)
    # visualize_trajectory("A-small", 0, 0)
