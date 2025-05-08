import numpy as np

from I_data_IO import *


# -------- Estimations, Calculations, Helper Functions -------- #

def estimate_home_location(df):
    # estimated home location = most frequent location during work hours (8 pm to 8 am)
    return (df
            [((0 <= df["t"]) & (df["t"] <= 16)) | ((40 <= df["t"]) & (df["t"] < 48))]
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


def compute_timedelta(group):
    timedelta = []

    for i in range(len(group)):
        if i == 0:
            timedelta.append(0)  # First row should be 0 (no previous timestamp)
        elif group.loc[i - 1, "is_recorded"] == 0:
            timedelta.append(1 + timedelta[i - 1])  # Accumulate previous delta
        else:
            timedelta.append(1)  # Reset the counter if is_recorded == 1

    group["timedelta"] = timedelta
    return group


def compute_timedelta_since_last_movement(group):
    timedelta = []

    for i in range(len(group)):
        if i == 0:
            timedelta.append(0)  # first row should be 0 (no previous timestamp)
        elif ((group.loc[i - 1, "is_recorded"] == 0) |
              ((group.loc[i, "x"] == group.loc[i - 1, "x"]) & (group.loc[i, "y"] == group.loc[i - 1, "y"]))):
            timedelta.append(1 + timedelta[i - 1])  # accumulate previous delta
        else:
            timedelta.append(1)  # reset the counter if is_recorded == 1 and movement was detected

    group["timedelta_since_last_movement"] = timedelta
    return group

# -------- Data Preprocessing -------- #

def data_preprocessing_user_trajectories(user_data, estimated_home, estimated_work):
    user_data = user_data.copy()

    # determine the number of unique users, days, and timestamps in the dataset
    num_users = user_data["uid"].nunique()
    num_days = user_data["d"].nunique()
    num_timestamps = user_data["t"].nunique()

    # create a dataframe containing all combinations of user_id, day, timestep within the respective limits
    all_possible_combinations = pd.DataFrame({
        "uid": np.repeat(np.arange(0, num_users), num_days * num_timestamps),
        "d": np.tile(np.repeat(np.arange(0, num_days), num_timestamps), num_users),
        "t": np.tile(np.tile(np.arange(0, num_timestamps), num_days), num_users)
    })

    # merge extended data with original user data
    user_data_extended = pd.merge(
        all_possible_combinations,
        user_data,
        on=["uid", "d", "t"],
        how="left",
        indicator=True  # adds a column "_merge" to indicate the source of each row
    )

    # mask column that is "1" if the (user_id, d, t) existed in original user_data, drop the merge indicator column
    user_data_extended["is_recorded"] = (user_data_extended["_merge"] == "both").astype(int)
    user_data_extended.drop(columns=["_merge"], inplace=True)

    # interpolate by using forward fill
    # user_data_extended[["x", "y"]] = user_data_extended.groupby("uid")[["x", "y"]].ffill().bfill().astype(int)

    # calculate cell_id
    user_data_extended["cell_id"] = (user_data_extended["x"] + (user_data_extended["y"] - 1) * 200).astype(int)

    # weekday
    user_data_extended["weekday"] = user_data_extended["d"] % 7

    # weekend
    user_data_extended["weekend"] = (
                (user_data_extended["weekday"] == 0) | (user_data_extended["weekday"] == 6)).astype(int)

    # daytime: 1, nighttime: 0 (= time between 7am and 7pm)
    user_data_extended["daytime"] = ((14 <= user_data_extended["t"]) & (user_data_extended["t"] < 38)).astype(int)

    # timedelta
    user_data_extended = user_data_extended.groupby("uid", group_keys=False).apply(compute_timedelta)

    # euclidean distance to last position
    user_data_extended["distance_to_last_position"] = calculate_euclidean_distance(
        user_data_extended["x"], user_data_extended["y"], user_data_extended["x"].shift(1).fillna(0),
        user_data_extended["y"].shift(1).fillna(0))

    # distance from estimated home location
    user_data_extended["distance_from_home"] = calculate_euclidean_distance(
        user_data_extended["x"], user_data_extended["y"], estimated_home["x"], estimated_home["y"])

    # distance from estimated work location
    user_data_extended["distance_from_work"] = calculate_euclidean_distance(
        user_data_extended["x"], user_data_extended["y"], estimated_work["x"], estimated_work["y"])

    # home: 0, work: 1, else/default: 2 (initialize with default value and apply masks)
    user_data_extended["currently_at"] = 2
    user_data_extended.loc[(user_data_extended["x"] == estimated_home["x"]) & (
                user_data_extended["y"] == estimated_home["y"]), "currently_at"] = 0
    user_data_extended.loc[(user_data_extended["x"] == estimated_work["x"]) & (
                user_data_extended["y"] == estimated_work["y"]), "currently_at"] = 1

    return user_data_extended


def data_preprocessing_static_graph(poi_data, user_data, estimated_home, estimated_work):
    poi_data = poi_data.copy()
    user_data = user_data.copy()

    # calculate cell_id
    poi_data["cell_id"] = poi_data["x"] + (poi_data["y"] - 1) * 200
    user_data["cell_id"] = user_data["x"] + (user_data["y"] - 1) * 200

    # create a dataframe containing all 40000 cells
    static_nodes = pd.DataFrame({
        "cell_id": np.arange(1, 40001),
        "x": np.tile(np.arange(1, 201), 200),  # Repeat 1 to 200 200 times for each x
        "y": np.repeat(np.arange(1, 201), 200)  # Repeat 1 to 200 for each y
    })

    poi_features = (
        poi_data
        .pivot_table(index="cell_id",
                     columns="category",
                     values="POI_count",
                     aggfunc="sum",
                     fill_value=0)
        .reset_index()
        .rename(columns=lambda x: f"poi_cat_{x}" if x != "cell_id" else "cell_id")
    )

    # distance from home, distance from work -> must be calculated dynamically
    # user_data["distance_from_home"] = calculate_euclidean_distance(user_data["x"], user_data["y"], estimated_home["x"], estimated_home["y"])
    # user_data["distance_from_work"] = calculate_euclidean_distance(user_data["x"], user_data["y"], estimated_work["x"], estimated_work["y"])

    # timedelta since last movement
    user_data = user_data.groupby("uid", group_keys=False).apply(compute_timedelta_since_last_movement)

    avg_dwell_time = (

    )

    visitor_count = (
        user_data
        .groupby("cell_id")
        .size()
        .pipe(lambda count: np.log1p(count) / np.log1p(count).max())
        .reset_index(name="visitor_count")
    )

    # merge everything together into one dataset
    merged = pd.merge(static_nodes, poi_features, on="cell_id", how="left")
    merged = pd.merge(merged, avg_dwell_time, on="cell_id", how="left")
    merged = pd.merge(merged, visitor_count, on="cell_id", how="left")

    return merged.fillna(0)

if __name__ == "__main__":
    RAW_USER_DATA_PATH = "./data/cityA-dataset.csv"
    user_data = load_csv_file(RAW_USER_DATA_PATH)

    RAW_POI_DATA_PATH = "./data/POIdata_cityA.csv"
    poi_data = load_csv_file(RAW_POI_DATA_PATH)

    home = estimate_home_location(user_data)
    work = estimate_work_location(user_data)

    preprocessed_user_data = data_preprocessing_user_trajectories(user_data, home, work)
    preprocessed_static_data = data_preprocessing_static_graph(poi_data, preprocessed_user_data, home, work)

    PREPROCESSED_TRAJECTORY_DATA_PATH = "./data/cityA-trajectory-dataset-preprocessed.csv"
    store_csv_file(PREPROCESSED_TRAJECTORY_DATA_PATH, preprocessed_user_data)

    PREPROCESSED_DATA_PATH = "./data/cityA-POIs-preprocessed.csv"
    store_csv_file(PREPROCESSED_DATA_PATH, preprocessed_static_data)
