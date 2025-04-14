import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def create_grid_plot(min_x, max_x, min_y, max_y):
    plot_dimensions_x = (max_x - min_x) / 2
    plot_dimensions_y = (max_y - min_y) / 2
    figure, axes = plt.subplots(figsize=(10 if plot_dimensions_x < 10 else plot_dimensions_x,
                                         10 if plot_dimensions_y < 10 else plot_dimensions_y))
    axes.set_xlabel("X", fontsize=12)
    axes.set_ylabel("Y", fontsize=12)

    axes.set_xticks(np.arange(min_x, max_x + 1, 1))
    axes.set_xticks(np.arange(-0.5, 200, 1), minor=True)
    axes.set_yticks(np.arange(min_y, max_y + 1, 1))
    axes.set_yticks(np.arange(-0.5, 200, 1), minor=True)
    axes.grid(which="minor")
    axes.tick_params(which="minor", size=0)

    axes.set_xlim(min_x - 0.5, max_x + 0.5)
    axes.set_ylim(min_y - 0.5, max_y + 0.5)
    axes.set_aspect("equal")

    return figure, axes


def create_single_trajectory_gif(df):
    NUM_FRAMES = 48
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    # min-max of the coordinate values
    min_x, max_x = df["x"].min() - 2, df["x"].max() + 2
    min_y, max_y = df["y"].min() - 2, df["y"].max() + 2

    # Set up figure and axes
    fig, ax = create_grid_plot(min_x, max_x, min_y, max_y)
    ax.set_title(label=f"Trajectory uid={UID} on day {DAY}", fontsize=21, fontweight="bold")

    # Plot elements
    line, = ax.plot([], [], alpha=0.7)
    scat_past = ax.scatter([], [], zorder=2)
    scat_current = ax.scatter([], [], color="red", zorder=3)
    quiver = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, width=0.005, color="blue")
    time_text = ax.annotate(
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

    # Create and save animation
    anim = FuncAnimation(fig, animate, frames=NUM_FRAMES + 1, interval=500, repeat=False)
    anim.save(f"./output/trajectory_0_day{DAY}.gif", writer="Pillow", dpi=80)


def create_real_vs_predicted_trajectory_gif(df):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    NUM_FRAMES = 48
    DAY = df["d"].iloc[0]
    UID = df["uid"].iloc[0]

    min_x = min(df["x_real"].min(), df["x_pred"].min()) - 2
    max_x = max(df["x_real"].max(), df["x_pred"].max()) + 2
    min_y = min(df["y_real"].min(), df["y_pred"].min()) - 2
    max_y = max(df["y_real"].max(), df["y_pred"].max()) + 2

    fig, ax = create_grid_plot(min_x, max_x, min_y, max_y)
    ax.set_title(label=f"Real vs Predicted Trajectory uid={UID} on day {DAY}", fontsize=20, fontweight="bold")

    line_real, = ax.plot([], [], color="blue", label="Real", alpha=0.7)
    line_pred, = ax.plot([], [], color="orange", label="Prediction", alpha=0.7)

    scat_real = ax.scatter([], [], color="blue", zorder=2)
    scat_pred = ax.scatter([], [], color="orange", zorder=2)

    scat_current_real = ax.scatter([], [], color="red", zorder=3)
    scat_current_pred = ax.scatter([], [], color="red", zorder=3)

    quiver_real = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, width=0.005, color="blue")
    quiver_pred = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, width=0.005, color="orange")

    time_text = ax.annotate(
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
    ax.legend()
    plt.tight_layout()

    def animate(i):
        tmp = df[df["t"] <= i]

        # Update lines
        line_real.set_data(tmp["x_real"], tmp["y_real"])
        line_pred.set_data(tmp["x_pred"], tmp["y_pred"])

        # Past scatter points (excluding last one)
        if len(tmp) > 1:
            past_real = tmp.iloc[:-1][["x_real", "y_real"]].values
            past_pred = tmp.iloc[:-1][["x_pred", "y_pred"]].values
        else:
            past_real = np.empty((0, 2))
            past_pred = np.empty((0, 2))

        scat_real.set_offsets(past_real)
        scat_pred.set_offsets(past_pred)

        if not tmp.empty:
            # Red dot: last real and predicted points
            last_real = tmp.iloc[-1][["x_real", "y_real"]].values.reshape(1, -1)
            last_pred = tmp.iloc[-1][["x_pred", "y_pred"]].values.reshape(1, -1)
            scat_current_real.set_offsets(last_real)
            scat_current_pred.set_offsets(last_pred)

            # Quiver logic
            if len(tmp) > 1:
                # Real
                x_real_last, y_real_last = tmp.iloc[-1][["x_real", "y_real"]]
                x_real_prev, y_real_prev = tmp.iloc[-2][["x_real", "y_real"]]
                dx_real, dy_real = x_real_last - x_real_prev, y_real_last - y_real_prev
                quiver_real.set_offsets([x_real_prev, y_real_prev])
                quiver_real.set_UVC(dx_real, dy_real)

                # Predicted
                x_pred_last, y_pred_last = tmp.iloc[-1][["x_pred", "y_pred"]]
                x_pred_prev, y_pred_prev = tmp.iloc[-2][["x_pred", "y_pred"]]
                dx_pred, dy_pred = x_pred_last - x_pred_prev, y_pred_last - y_pred_prev
                quiver_pred.set_offsets([x_pred_prev, y_pred_prev])
                quiver_pred.set_UVC(dx_pred, dy_pred)
            else:
                quiver_real.set_offsets([0, 0])
                quiver_real.set_UVC(0, 0)
                quiver_pred.set_offsets([0, 0])
                quiver_pred.set_UVC(0, 0)

            time_text.set_text(f"Time: {i}")
        else:
            scat_current_real.set_offsets(np.empty((0, 2)))
            scat_current_pred.set_offsets(np.empty((0, 2)))
            quiver_real.set_offsets([0, 0])
            quiver_real.set_UVC(0, 0)
            quiver_pred.set_offsets([0, 0])
            quiver_pred.set_UVC(0, 0)
            time_text.set_text(f"Time: {i} (no data)")

        return (
            scat_real, scat_pred, line_real, line_pred,
            scat_current_real, scat_current_pred,
            quiver_real, quiver_pred, time_text
        )

    anim = FuncAnimation(fig, animate, frames=NUM_FRAMES + 1, interval=500, repeat=False)
    anim.save(f"./output/real_vs_predicted_uid{UID}_day{DAY}.gif", writer="Pillow", dpi=80)
