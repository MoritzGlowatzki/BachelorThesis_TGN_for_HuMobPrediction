import numpy as np

from I_data_IO import *


# -------- Estimations, Calculations, Helper Functions -------- #

def estimate_home_location(df):
    # estimated home location = most frequent location during work hours (10 pm to 6 am)
    return (df
            [((0 <= df["t"]) & (df["t"] <= 12)) | ((44 <= df["t"]) & (df["t"] < 48))]
            [["x", "y"]]
            .agg(lambda location: location.mode().iloc[0]))

def estimate_work_location(df):
    # estimated work location = most frequent location during work hours (9 am to 5 pm)
    return (df
            [(18 <= df["t"]) & (df["t"] < 34)]
            [["x", "y"]]
            .agg(lambda location: location.mode().iloc[0]))

def calculate_average_number_of_movements_per_day(df, uid):
    return df[df["uid" == uid]].size().mean()

def calculate_average_travel_distance_per_day(df, uid):
    if "distance_to_last_position" in df.columns:
        df["distance_to_last_position"] = calculate_euclidean_distance(df["x"], df["y"], df["x"].shift(1).fillna(0),
                                                                       df["y"].shift(1).fillna(0))
    return df[df["uid" == uid]].groupby("d")["distance_to_last_position"].mean()

def calculate_euclidean_distance(x_1, y_1, x_2, y_2):
    # helper function for calculating the euclidean distance
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


# -------- Data Preprocessing Pipeline -------- #

def data_preprocessing(data):
    # sin-cos-transformations of "day"
    data["d_sin"] = np.sin(data["d"] / 7)
    data["d_cos"] = np.cos(data["d"] / 7)

    # sin-cos-transformations of "time"
    data["t_sin"] = np.sin(data["t"] / 48)
    data["t_cos"] = np.cos(data["t"] / 48)

    # weekday
    data["weekday"] = data["d"] % 7

    # weekend
    data["weekend"] = (data["weekday"] == 0) | (data["weekday"] == 6)

    # daytime: 1, nighttime: 0 (= time between 7am and 7pm)
    data["daytime"] = (14 <= data["t"]) & (data["t"] < 38)

    # am: 1, pm: 0
    data["am"] = (0 <= data["t"]) & (data["t"] < 24)

    # timedelta
    data["timedelta"] = data["t"] - data["t"].shift(1).fillna(0)

    # euclidean distance to last position
    data["distance_to_last_position"] = calculate_euclidean_distance(data["x"], data["y"], data["x"].shift(1).fillna(0),
                                                                     data["y"].shift(1).fillna(0))

    # velocity = (2 * euclidean_distance to last position) / hour (unit: grid cell distance per hour)
    data["velocity"] = 2 * data["distance_to_last_position"] / 1

    # distance from home
    home = estimate_home_location(data)
    data["distance_from_home"] = calculate_euclidean_distance(data["x"], data["y"], home["x"], home["y"])

    # distance from work
    work = estimate_work_location(data)
    data["distance_from_work"] = calculate_euclidean_distance(data["x"], data["y"], work["x"], work["y"])

    # home: 0, work: 1, else: 2
    # Initialize with default value and apply masks
    data["currently_at"] = 2
    data.loc[(data["x"] == home["x"]) & (data["y"] == home["y"]), "currently_at"] = 0
    data.loc[(data["x"] == work["x"]) & (data["y"] == work["y"]), "currently_at"] = 1

    return data


if __name__ == "__main__":
    RAW_DATA_PATH = "./data/cityA-dataset-small.csv"
    data = load_csv_file(RAW_DATA_PATH)

    preprocessed_data = data_preprocessing(data)

    PREPROCESSED_DATA_PATH = "./data/cityA-dataset-small-preprocessed.csv"
    store_csv_file(PREPROCESSED_DATA_PATH, data)
