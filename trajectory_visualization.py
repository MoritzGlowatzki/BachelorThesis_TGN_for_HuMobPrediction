from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# -------- Preliminaries -------- #
NUM_FRAMES = 48
SUPPORTED_MODES = {
    "both": [("x", "x_pred"), ("y", "y_pred")],
    "real": [("x",), ("y",)],
    "pred": [("x_pred",), ("y_pred",)]
}


# -------- Helper Functions -------- #
def find_min_max_coordinates(df, mode: Literal["both", "real", "pred"]):
    x_cols, y_cols = SUPPORTED_MODES[mode]

    min_x = int(min(df[col].min() for col in x_cols)) - 2
    max_x = int(max(df[col].max() for col in x_cols)) + 2
    min_y = int(min(df[col].min() for col in y_cols)) - 2
    max_y = int(max(df[col].max() for col in y_cols)) + 2

    return min_x, max_x, min_y, max_y


def create_grid_plot(min_x, max_x, min_y, max_y, nrows: int = 1, ncols: int = 1):
    # Compute range of x and y
    plot_dimension_x = (max_x - min_x + 1) / 2
    plot_dimension_y = (max_y - min_y + 1) / 2

    # Calculate size per subplot (base values between 8 and 16 inches)
    fig_width = max(8, min(16, plot_dimension_x)) * 0.75 * ncols
    fig_height = max(8, min(16, plot_dimension_y)) * 0.75 * nrows

    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    for ax in np.atleast_1d(axes):
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)

        ax.set_xticks(np.arange(min_x, max_x + 1, 1))
        ax.set_xticks(np.arange(-0.5, 200, 1), minor=True)
        ax.set_yticks(np.arange(min_y, max_y + 1, 1))
        ax.set_yticks(np.arange(-0.5, 200, 1), minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", size=0)

        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.set_aspect("equal")

    return figure, axes


def setup_time_text(ax):
    return ax.annotate(
        "",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=14,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )


def create_and_safe_animation(fig, animate, path):
    anim = FuncAnimation(fig, animate, frames=NUM_FRAMES + 1, interval=500, repeat=False)
    anim.save(path, writer="Pillow", dpi=80)


# -------- Trajectory Visualization (GIF) -------- #
def create_single_trajectory_gif(df, mode: Literal["real", "pred"]):
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    fig, axis = create_grid_plot(*find_min_max_coordinates(df, mode=mode))

    if mode == "real":
        axis.set_title(label=f"Predicted Trajectory uid={UID} on day {DAY}", fontsize=21, fontweight="bold")
        color = "blue"
    else:
        axis.set_title(label=f"Trajectory uid={UID} on day {DAY}", fontsize=21, fontweight="bold")
        color = "orange"

    # Plot elements
    line, = axis.plot([], [], alpha=0.7)
    scat_past = axis.scatter([], [], zorder=2)
    scat_current = axis.scatter([], [], color="red", zorder=3)
    quiver = axis.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, width=0.005, color=color)
    time_text = setup_time_text(axis)
    plt.tight_layout()

    def animate(i):
        tmp = df[df["t"] <= i]
        current_frame_data = df[df["t"] == i]

        if not current_frame_data.empty:
            # Get past and current points
            if len(tmp) > 1:
                past_points = tmp.iloc[:-1][["x", "y"]].values
                last_point = tmp.iloc[-1][["x", "y"]].values
            else:
                past_points = np.empty((0, 2))
                last_point = tmp.iloc[-1][["x", "y"]].values

            # Update scatter plots
            scat_past.set_offsets(past_points)
            scat_current.set_offsets([last_point])

            # Update line
            line.set_data(tmp["x"], tmp["y"])

            # Update quiver
            if len(tmp) > 1:
                x_last, y_last = tmp.iloc[-1][["x", "y"]]
                x_prev, y_prev = tmp.iloc[-2][["x", "y"]]
                dx, dy = x_last - x_prev, y_last - y_prev

                quiver.set_offsets([x_prev, y_prev])
                quiver.set_UVC(dx, dy)
            else:
                quiver.set_offsets([0, 0])
                quiver.set_UVC(0, 0)

            # Update time label
            time_text.set_text(f"Time: {i}")
        else:
            # No update, hide quiver and current point
            quiver.set_offsets([0, 0])
            quiver.set_UVC(0, 0)
            time_text.set_text(f"Time: {i} (no data)")

        return scat_past, scat_current, line, quiver, time_text

    create_and_safe_animation(fig, animate, path=f"./output/trajectory_0_day{DAY}.gif")


def create_combined_trajectory_gif(df):
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    # Create one figure per subplot with consistent grid setup
    fig, axes = create_grid_plot(*find_min_max_coordinates(df, mode="both"), nrows=1, ncols=3)
    fig.suptitle(t=f"Real vs Predicted Trajectory uid={UID} on day {DAY}", fontsize=20, fontweight="bold")

    axes[0].set_title("predicted")
    axes[1].set_title("combined")
    axes[2].set_title("real")

    # Plot elements
    line_pred_0, = axes[0].plot([], [], color="orange")
    scat_pred_0 = axes[0].scatter([], [], color="orange")
    current_pred_0 = axes[0].scatter([], [], color="red", zorder=3)

    line_real_2, = axes[2].plot([], [], color="blue")
    scat_real_2 = axes[2].scatter([], [], color="blue")
    current_real_2 = axes[2].scatter([], [], color="red", zorder=3)

    line_real_1, = axes[1].plot([], [], color="blue", label="Real")
    line_pred_1, = axes[1].plot([], [], color="orange", label="Prediction")
    scat_real_1 = axes[1].scatter([], [], color="blue")
    scat_pred_1 = axes[1].scatter([], [], color="orange")
    current_real_1 = axes[1].scatter([], [], color="red", zorder=3)
    current_pred_1 = axes[1].scatter([], [], color="red", zorder=3)

    # Time labels
    time_texts = []
    for ax in axes:
        text = setup_time_text(ax)
        time_texts.append(text)

    axes[1].legend()
    plt.tight_layout()

    def animate(i):
        tmp = df[df["t"] <= i]
        current_frame_data = df[df["t"] == i]

        past_real = tmp[tmp["t"] < i][["x", "y"]].values
        past_pred = tmp[tmp["t"] < i][["x_pred", "y_pred"]].values

        # predicted subplot
        line_pred_0.set_data(tmp["x_pred"], tmp["y_pred"])
        scat_pred_0.set_offsets(past_pred)

        # real subplot
        line_real_2.set_data(tmp["x"], tmp["y"])
        scat_real_2.set_offsets(past_real)

        # combined subplot
        line_real_1.set_data(tmp["x"], tmp["y"])
        line_pred_1.set_data(tmp["x_pred"], tmp["y_pred"])
        scat_real_1.set_offsets(past_real)
        scat_pred_1.set_offsets(past_pred)

        # Always show current position
        if not current_frame_data.empty:
            current = current_frame_data.iloc[0]
            current_pred_0.set_offsets([[current["x_pred"], current["y_pred"]]])
            current_real_2.set_offsets([[current["x"], current["y"]]])
            current_real_1.set_offsets([[current["x"], current["y"]]])
            current_pred_1.set_offsets([[current["x_pred"], current["y_pred"]]])

            for text in time_texts:
                text.set_text(f"Time: {i}")
        else:
            # If no data for the current frame, still keep the red dots at previous positions
            last_pred = tmp.iloc[-1][["x_pred", "y_pred"]].values if not tmp.empty else np.array([np.nan, np.nan])
            last_real = tmp.iloc[-1][["x", "y"]].values if not tmp.empty else np.array([np.nan, np.nan])

            current_pred_0.set_offsets([last_pred])
            current_real_2.set_offsets([last_real])
            current_real_1.set_offsets([last_real])
            current_pred_1.set_offsets([last_pred])

            for text in time_texts:
                text.set_text(f"Time: {i} (no data)")

        return (
            line_pred_0, scat_pred_0, current_pred_0,
            line_real_2, scat_real_2, current_real_2,
            line_real_1, line_pred_1,
            scat_real_1, scat_pred_1,
            current_real_1, current_pred_1,
            *time_texts
        )

    create_and_safe_animation(fig, animate, path=f"./output/comparison_trajectory_uid{UID}_day{DAY}.gif")


# -------- Trajectory Visualization (PNG) -------- #
def create_single_trajectory_plot(df, mode: Literal["real", "pred"]):
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    fig, axis = create_grid_plot(*find_min_max_coordinates(df, mode=mode))

    # Plot trajectory
    if mode == "real":
        axis.plot(df["x"], df["y"], color="blue")
        axis.scatter(df["x"], df["y"], color="blue")
    else:
        axis.plot(df["x_pred"], df["y_pred"], color="orange")
        axis.scatter(df["x_pred"], df["y_pred"], color="orange")

    fig.suptitle(f"Trajectory Comparison uid={UID} on day {DAY}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"./output/comparison_trajectory_uid={UID}_day{DAY}.png")
    plt.show()


def create_combined_trajectory_plot(df):
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    fig, axes = create_grid_plot(*find_min_max_coordinates(df, mode="both"), nrows=1, ncols=3)

    # Subplot 1: Only Prediction
    axes[0].plot(df["x_pred"], df["y_pred"], color="orange", label="Prediction")
    axes[0].scatter(df["x_pred"], df["y_pred"], color="orange")
    axes[0].set_title("predicted")

    # Subplot 2: Both Real & Prediction
    axes[1].plot(df["x"], df["y"], color="blue", label="Real")
    axes[1].scatter(df["x"], df["y"], color="blue")
    axes[1].plot(df["x_pred"], df["y_pred"], color="orange", label="Prediction")
    axes[1].scatter(df["x_pred"], df["y_pred"], color="orange")
    axes[1].set_title("combined")
    axes[1].legend()

    # Subplot 3: Only Real
    axes[2].plot(df["x"], df["y"], color="blue", label="Real")
    axes[2].scatter(df["x"], df["y"], color="blue")
    axes[2].set_title("real")

    fig.suptitle(f"Trajectory Comparison uid={UID} on day {DAY}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"./output/comparison_trajectory_uid={UID}_day{DAY}.png")
    plt.show()
