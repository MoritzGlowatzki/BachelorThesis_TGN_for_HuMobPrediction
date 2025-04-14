import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def create_grid_plot(min_x, max_x, min_y, max_y):
    plot_dimensions_x = (max_x - min_x) / 2
    plot_dimensions_y = (max_y - min_y) / 2
    figure, axes = plt.subplots(figsize=(10 if plot_dimensions_x < 10 else plot_dimensions_x,
                                         10 if plot_dimensions_y < 10 else plot_dimensions_y))
    axes.set_xlabel('X', fontsize=12)
    axes.set_ylabel('Y', fontsize=12)

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
    DAY = df["d"][0]
    UID = df["uid"][0]

    # min-max of the coordinate values
    min_x, max_x = df['x'].min() - 2, df['x'].max() + 2
    min_y, max_y = df['y'].min() - 2, df['y'].max() + 2

    # Set up figure and axes
    fig, ax = create_grid_plot(min_x, max_x, min_y, max_y)
    ax.set_title(label=f'Trajectory uid={UID} on day {DAY}', fontsize=21, fontweight='bold')

    # Plot elements
    line, = ax.plot([], [], alpha=0.7)
    scat_past = ax.scatter([], [], zorder=2)
    scat_current = ax.scatter([], [], color="red", zorder=3)
    quiver = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1, width=0.005)
    time_text = ax.annotate(
        '',
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(10, -10),
        textcoords='offset points',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.tight_layout()

    def animate(i):
        tmp = df[df['t'] <= i]
        current_frame_data = df[df['t'] == i]

        if not current_frame_data.empty:
            # Get past and current points
            if len(tmp) > 1:
                past_points = tmp.iloc[:-1][['x', 'y']].values
                last_point = tmp.iloc[-1][['x', 'y']].values
            else:
                past_points = np.empty((0, 2))
                last_point = tmp.iloc[-1][['x', 'y']].values

            # Update scatter plots
            scat_past.set_offsets(past_points)
            scat_current.set_offsets([last_point])

            # Update line
            line.set_data(tmp['x'], tmp['y'])

            # Update quiver
            if len(tmp) > 1:
                x_last, y_last = tmp.iloc[-1][['x', 'y']]
                x_prev, y_prev = tmp.iloc[-2][['x', 'y']]
                dx, dy = x_last - x_prev, y_last - y_prev

                quiver.set_offsets([x_prev, y_prev])
                quiver.set_UVC(dx, dy)
            else:
                quiver.set_offsets([0, 0])
                quiver.set_UVC(0, 0)

            # Update time label
            time_text.set_text(f'Time: {i}')
        else:
            # No update, hide quiver and current point
            quiver.set_offsets([0, 0])
            quiver.set_UVC(0, 0)
            time_text.set_text(f'Time: {i} (no data)')

        return scat_past, scat_current, line, quiver, time_text

    # Create and save animation
    anim = FuncAnimation(fig, animate, frames=NUM_FRAMES + 1, interval=500, repeat=False)
    anim.save(f'./output/trajectory_0_day{DAY}.gif', writer='Pillow', dpi=80)


def create_real_vs_predicted_trajectory_gif(df):
    # TODO
    pass
