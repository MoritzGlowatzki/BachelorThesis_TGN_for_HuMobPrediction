from matplotlib import pyplot as plt
from tqdm import tqdm

from I_data_IO import *

# -------- Preliminaries -------- #
data_paths = [
    "./data/cityA-dataset.csv",
    "./data/cityB-dataset.csv",
    "./data/cityC-dataset.csv",
    "./data/cityD-dataset.csv"
]
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
for path, label, color in tqdm(zip(data_paths, labels_map, colors_map), total=len(data_paths)):
    data = load_csv_file(path)
    counts = data.groupby("d").size()
    counts.plot(style=".-", label=label, color=color)

# -------- Log scale for y-axis -------- #
plt.yscale("log")
plt.xlabel("Day")
plt.ylabel("Number of records (log scale)")
plt.legend()
plt.title("Daily Data Records per City")
plt.tight_layout()

# -------- Show and save plot -------- #
plt.savefig("./output/histogram_data_records_per_day.png")
plt.show()
