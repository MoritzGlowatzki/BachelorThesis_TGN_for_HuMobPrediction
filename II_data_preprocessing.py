from math import cos
from math import sin

import numpy as np

from I_data_IO import *

RAW_DATA_PATH = "./data/cityA-dataset-small.csv"
data = load_csv_file(RAW_DATA_PATH)


def estimate_home_location(uid):
    # estimated home location = most frequent location during work hours (10 pm to 6 am)
    return (data
            .filter(uid)
            .filter(0 <= data["t"] <= 12 | 44 <= data["t"] < 48)
            [["x", "y"]]
            .agg(lambda location: location.mode().iloc[0]))


def estimate_work_location(uid):
    # estimated work location = most frequent location during work hours (9 am to 5 pm)
    return (data
            .filter(uid)
            .filter(18 <= data["t"] < 34)
            [["x", "y"]]
            .agg(lambda location: location.mode().iloc[0]))


# helper function for calculating the euclidean distance
def calculate_euclidean_distance(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


home = estimate_home_location(data["uid"])
# work = estimate_work_location(data["uid"])

# sin-cos-transformations of "day"
data["d_sin"] = sin(data["d"] / 7)
data["d_cos"] = cos(data["d"] / 7)

# sin-cos-transformations of "time"
data["t_sin"] = sin(data["t"] / 48)
data["t_cos"] = cos(data["t"] / 48)

# weekday
data["weekday"] = data["d"] % 7

# weekend
data["weekend"] = (data["weekday"] == 0 | data["weekday"] == 6)

# daytime: 1, nighttime: 0 (= time between 7am and 7pm)
data["daytime"] = (14 <= data["t"] < 38)

# am: 1, pm: 0
data["am"] = (0 <= data["t"] < 24)

# timedelta
data["timedelta"] -= data["timedelta"].shift(1).fillna(0)

# euclidean distance to last position
data["distance_to_last_position"] = calculate_euclidean_distance(data["x"], data["y"], data["x"].shift(1).fillna(0),
                                                                 data["y"].shift(1).fillna(0))

# velocity = (2 * euclidean_distance to last position) / hour (unit: grid cell distance per hour)
data["velocity"] = 2 * data["distance_to_last_position"] / 1

# distance from home
data["distance_from_home"] = calculate_euclidean_distance(data["x"], data["y"], home["x"], home["y"])

# distance from work
# data["distance_from_work"] = calculate_euclidean_distance(data["x"], data["y"], work["x"], work["y"])

# home: 0, work: 1, else: 2
data["currently_at"] = (
    0 if ((data["x"] == home["x"]) & (data["y"] == home["y"]))
    # else 1 if ((data["x"] == work["x"]) & (data["y"] == work["y"]))
    else 2
)

# average number of movements per day
average_number_of_movements = data.groupby("uid", "d").size().mean()

# average travel distance per day
average_travel_distance = data.groupby("uid", "d")["distance_to_last_position"].mean()

PREPROCESSED_DATA_PATH = "./data/cityA-dataset-small-preprocessed.csv"
store_csv_file(RAW_DATA_PATH, data)
