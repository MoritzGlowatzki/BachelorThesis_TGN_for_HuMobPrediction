import numpy as np
from tqdm import tqdm

from _I_data_IO import *


# -------- Estimations, Calculations, Helper Functions -------- #

def estimate_home_location(df):
    # estimated home = most frequent location from 8 PM to 8 AM (t in [0–16] and [40–48])
    filtered_df = df[((0 <= df["t"]) & (df["t"] <= 16)) | ((40 <= df["t"]) & (df["t"] < 48))]

    if not filtered_df.empty:
        coords = filtered_df.groupby(["x", "y"]).size().idxmax()
    else:
        # Fallback: use most frequent location in the entire data
        coords = df.groupby(["x", "y"]).size().idxmax()

    return {
        "home_x": coords[0], "home_y": coords[1],
        "home_cell_id": calculate_cell_id({"x": coords[0], "y": coords[1]})
    }

def estimate_work_location(df):
    # estimated work = most frequent location from 9 AM to 7 PM (t in [18–38])
    filtered_df = df[(18 <= df["t"]) & (df["t"] < 38)]

    if not filtered_df.empty:
        coords = filtered_df.groupby(["x", "y"]).size().idxmax()
    else:
        # Fallback: use second most frequent location in the entire data (if existent, otherwise use most frequent location)
        freq = df.groupby(["x", "y"]).size().sort_values(ascending=False)
        coords = freq.index[1] if len(freq) > 1 else freq.index[0]

    return {
        "work_x": coords[0], "work_y": coords[1],
        "work_cell_id": calculate_cell_id({"x": coords[0], "y": coords[1]})
    }

def calculate_cell_id(coordinate):
    return int(coordinate["x"]) + (int(coordinate["y"]) - 1) * 200

def calculate_average_number_of_movements_per_day(group):
    return group.groupby("d").size().mean()

def calculate_average_travel_distance_per_day(group):
    return group.groupby("d")["distance_to_last_position"].sum().mean()

def calculate_euclidean_distance(x1, y1, x2, y2):
    # helper function for calculating the euclidean distance
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_timedelta_since_last_movement(group):
    group = group.reset_index(drop=True)
    delta = [0]

    for i in range(1, len(group)):
        if (group.loc[i, "is_recorded"] == 0) or (group.loc[i - 1, "cell_id"] == group.loc[i, "cell_id"]):
            delta.append(delta[-1] + 1)
        else:
            delta.append(0)

    group["timedelta_since_last_movement"] = delta
    return group

def calculate_timedelta_since_last_movement_fast(group):
    group = group.reset_index(drop=True)

    # Mark rows where movement (= actual record and cell_id change) is detected (True if movement, else False)
    movement_detected = (group["is_recorded"] == 1) & (group["cell_id"] != group["cell_id"].shift())
    movement_id = movement_detected.cumsum()

    # Get cumulative count within each movement
    group["timedelta_since_last_movement"] = group.groupby(movement_id).cumcount()
    return group

def calculate_timedelta_since_last_observation(group):
    group = group.reset_index(drop=True)
    timedelta = []

    # Find index of the first recorded observation
    first_recorded_idx = group[group["is_recorded"] == 1].index.min()

    for i in range(len(group)):
        if i < first_recorded_idx:
            # Count backwards (negative timedelta) until first recording
            timedelta.append(i - first_recorded_idx)
        elif i == first_recorded_idx:
            # Zero at the first recorded observation
            timedelta.append(0)
        elif group.loc[i - 1, "is_recorded"] == 0:
            # Accumulate from last
            timedelta.append(timedelta[i - 1] + 1)
        else:
            # Reset to 1 after a recorded event
            timedelta.append(1)

    group["timedelta_since_last_observation"] = timedelta
    return group

def calculate_timedelta_since_last_observation_fast(group):
    group = group.reset_index(drop=True)
    first_record_idx = group.index[group["is_recorded"] == 1].min()

    last_record_idx = (
        group.index.where(group["is_recorded"] == 1)
        .to_series()
        .ffill()
        .shift(1)
        .fillna(first_record_idx)
    )

    group["timedelta_since_last_observation"] = group.index.to_numpy() - last_record_idx.to_numpy()
    return group

def add_stay_id(group):
    group = group.reset_index(drop=True)
    group["stay_id"] = (group["cell_id"] != group["cell_id"].shift()).cumsum() - 1
    return group

def downcast_dataframe(df):
    # convert each column datatype to the smallest possible that can still hold all its values
    for col in df.select_dtypes(include=["int", "int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

# -------- Data Preprocessing -------- #

def data_preprocessing_user_trajectories(user_data):
    traj_data = user_data.copy()

    # determine the number of unique users, days, and timestamps in the dataset
    num_users = traj_data["uid"].nunique()
    num_days = traj_data["d"].nunique()
    num_timestamps = traj_data["t"].nunique()

    # create a dataframe containing all combinations of user_id, day, timestep within the respective limits
    all_possible_combinations = pd.DataFrame({
        "uid": np.repeat(np.arange(0, num_users), num_days * num_timestamps),
        "d": np.tile(np.repeat(np.arange(0, num_days), num_timestamps), num_users),
        "t": np.tile(np.tile(np.arange(0, num_timestamps), num_days), num_users)
    })

    # merge extended data with original user data
    traj_data_extended = pd.merge(
        all_possible_combinations,
        traj_data,
        on=["uid", "d", "t"],
        how="left",
        indicator=True  # adds a column "_merge" to indicate the source of each row
    )

    # mask column that is "1" if the (user_id, d, t) existed in original traj_data, drop the merge indicator column
    traj_data_extended["is_recorded"] = (traj_data_extended["_merge"] == "both").astype(int)
    traj_data_extended.drop(columns=["_merge"], inplace=True)

    # remove day 27 as advised in "YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories" (Yabe et al. 2024)
    traj_data_extended = traj_data_extended[traj_data_extended["d"] != 26]

    # interpolate by using forward fill
    traj_data_extended[["x", "y"]] = traj_data_extended.groupby("uid")[["x", "y"]].ffill().bfill().astype(int)

    # calculate cell_id
    traj_data_extended["cell_id"] = ((traj_data_extended["x"] + (traj_data_extended["y"] - 1) * 200).astype(int))

    # calculate timestamp
    traj_data_extended["timestamp"] = (traj_data_extended["d"] * 48 + traj_data_extended["t"]).astype(int)

    # Sine-cosine encoding of 'day' (0–6) and 'time' (0–47)
    traj_data_extended["d_sin"] = np.sin(2 * np.pi * traj_data_extended["d"] / 7)
    traj_data_extended["d_cos"] = np.cos(2 * np.pi * traj_data_extended["d"] / 7)
    traj_data_extended["t_sin"] = np.sin(2 * np.pi * traj_data_extended["t"] / 48)
    traj_data_extended["t_cos"] = np.cos(2 * np.pi * traj_data_extended["t"] / 48)

    # weekday
    traj_data_extended["weekday"] = traj_data_extended["d"] % 7
    traj_data_extended = pd.get_dummies(traj_data_extended, columns=["weekday"], prefix="wd")

    # weekend
    traj_data_extended["weekend"] = (
            (traj_data_extended["wd_0"] == 1) | (traj_data_extended["wd_6"] == 1))

    # daytime: 1, nighttime: 0 (= time between 7am and 7pm)
    traj_data_extended["daytime"] = ((14 <= traj_data_extended["t"]) & (traj_data_extended["t"] < 38))

    # timedelta_since_last_movement and timedelta_since_last_observation
    processed_users = []
    for uid, group in tqdm(traj_data_extended.groupby("uid"), desc="Calculating trajectory stats"):
        # group = calculate_timedelta_since_last_movement(group)
        group = calculate_timedelta_since_last_movement_fast(group)
        # group = calculate_timedelta_since_last_observation(group)
        # group = calculate_timedelta_since_last_observation_fast(group)

        # distance from estimated home location
        home = estimate_home_location(group)
        group["home_x"] = home["home_x"]
        group["home_y"] = home["home_y"]
        group["distance_from_home"] = calculate_euclidean_distance(group["x"], group["y"], home["home_x"],
                                                                   home["home_y"])

        # distance from estimated work location
        work = estimate_work_location(group)
        group["work_x"] = work["work_x"]
        group["work_y"] = work["work_y"]
        group["distance_from_work"] = calculate_euclidean_distance(group["x"], group["y"], work["work_x"],
                                                                   work["work_y"])

        processed_users.append(group)

    traj_data_extended = pd.concat(processed_users, ignore_index=True)

    # euclidean distance to last position
    traj_data_extended["x_prev"] = traj_data_extended.groupby("uid")["x"].shift(1)
    traj_data_extended["y_prev"] = traj_data_extended.groupby("uid")["y"].shift(1)

    traj_data_extended["distance_to_last_position"] = calculate_euclidean_distance(
        traj_data_extended["x"], traj_data_extended["y"],
        traj_data_extended["x_prev"].bfill(), traj_data_extended["y_prev"].bfill())
    traj_data_extended.drop(columns=["x_prev", "y_prev"], inplace=True)

    # home: 0, work: 1, else/default: 2 (initialize with default value and apply masks)
    traj_data_extended["position_flag"] = 2
    traj_data_extended.loc[
        (traj_data_extended["x"] == traj_data_extended["home_x"]) &
        (traj_data_extended["y"] == traj_data_extended["home_y"]),
        "position_flag"] = 0
    traj_data_extended.loc[
        (traj_data_extended["x"] == traj_data_extended["work_x"]) &
        (traj_data_extended["y"] == traj_data_extended["work_y"]),
        "position_flag"] = 1

    # drop home_x, home_y, work_x and work_y
    traj_data_extended.drop(columns=["home_x", "home_y", "work_x", "work_y"], inplace=True)

    # Cast all columns except "distance_to_last_position", "distance_from_home", "distance_from_work" to int
    cols_to_int = traj_data_extended.columns.difference(
        ["distance_to_last_position", "distance_from_home", "distance_from_work"])
    traj_data_extended[cols_to_int] = traj_data_extended[cols_to_int].astype(int)

    return downcast_dataframe(traj_data_extended)

def data_preprocessing_user_info(user_data):
    user_info = user_data.copy()

    stats = []
    for uid, group in tqdm(user_info.groupby("uid"), desc="Calculating user stats"):
        home = estimate_home_location(group)
        work = estimate_work_location(group)
        stats.append({
            "uid": uid,
            "home_x": home["home_x"],
            "home_y": home["home_y"],
            "home_cell_id": home["home_cell_id"],
            "work_x": work["work_x"],
            "work_y": work["work_y"],
            "work_cell_id": work["work_cell_id"],
            "average_movements_per_day": calculate_average_number_of_movements_per_day(group),
            "average_travel_distance_per_day": calculate_average_travel_distance_per_day(group)
        })

    # build the DataFrame
    result = pd.DataFrame(stats)

    # Cast all columns except "average_movements_per_day" and "average_travel_distance_per_day" to int
    cols_to_int = result.columns.difference(["average_movements_per_day", "average_travel_distance_per_day"])
    result[cols_to_int] = result[cols_to_int].astype(int)

    return downcast_dataframe(result)

def data_preprocessing_static_graph(poi_data, user_data):
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

    # POI_feature count per category and total_POI_count
    poi_features = (
        poi_data
        .pivot_table(index="cell_id",
                     columns="category",
                     values="POI_count",
                     aggfunc="sum",
                     fill_value=0)
        .reset_index()
        .rename(columns=lambda x: f"POI_cat_{x}" if x != "cell_id" else "cell_id")
        .assign(total_POI_count=lambda df: df.drop(columns="cell_id").sum(axis=1).astype(int))
    )

    # average dwell time per cell
    avg_dwell_time = (
        user_data
        .groupby("uid", group_keys=False)
        [["uid", "timedelta_since_last_movement", "cell_id"]]
        .apply(add_stay_id)
        .groupby(["uid", "cell_id", "stay_id"], as_index=False)
        .agg(mean_stay_duration_per_user_per_cell=("timedelta_since_last_movement", "mean"))
        .groupby("cell_id", as_index=False)
        .agg(avg_dwell_time=("mean_stay_duration_per_user_per_cell", "mean"))
    )

    # normalized visitor count per cell
    visitor_count = (
        user_data
        .groupby("cell_id")
        .size()
        .pipe(lambda count: np.log1p(count) / np.log1p(count).max())
        .reset_index(name="visitor_count")
    )

    # merge everything together into one dataset
    merged = pd.merge(static_nodes, poi_features, on="cell_id", how="left")
    merged = pd.merge(merged, visitor_count, on="cell_id", how="left")
    merged = pd.merge(merged, avg_dwell_time, on="cell_id", how="left")
    merged = merged.fillna(0)

    # Cast all columns except "avg_dwell_time" to int
    cols_to_int = merged.columns.difference(["avg_dwell_time"])
    merged[cols_to_int] = merged[cols_to_int].astype(int)

    return downcast_dataframe(merged)


# -------- Data Checks -------- #

def check_for_new_cells_after_day_60(df, city_idx):
    visited_before = set(map(tuple, df.loc[df["d"] < 60, ["x", "y"]].values))
    visited_after = set(map(tuple, df.loc[df["d"] >= 60, ["x", "y"]].values))
    new_cells_after_60 = visited_after.difference(visited_before)
    total_visited = df.drop_duplicates(subset=["x", "y"]).shape[0]
    if new_cells_after_60:
        print(
            f"{len(new_cells_after_60)} out of {total_visited} cells in city {city_idx} visited after day 60 were new (not seen before).")
    else:
        print(f"No new cells visited in city {city_idx} after day 60 — all were already seen before.")

if __name__ == "__main__":
    for city_idx in ["B", "C", "D"]:  # "A", "B", "C", "D"
        print(f"Currently processing city: {city_idx}")

        print("Load city data ... ")
        RAW_CITY_DATA_PATH = f"./data/original/city{city_idx}-dataset.csv"
        user_data = load_csv_file(RAW_CITY_DATA_PATH)
        print("Finished loading city data.")

        # check_for_new_cells_after_day_60(user_data, city_idx)

        # for the 2024 challenge only days 1 to 60 were known
        if city_idx != "A":
            user_data = user_data[user_data["d"] < 60]

        print("Start processing trajectory data ... ")
        preprocessed_trajectory_data = data_preprocessing_user_trajectories(user_data)
        PREPROCESSED_TRAJECTORY_DATA_PATH = f"./data/raw/city{city_idx}-trajectory-dataset-preprocessed.csv"
        store_csv_file(PREPROCESSED_TRAJECTORY_DATA_PATH, preprocessed_trajectory_data)
        print("Finished processing trajectory data.")

        print("Load POI data ... ")
        RAW_POI_DATA_PATH = f"./data/original/POIdata_city{city_idx}.csv"
        poi_data = load_csv_file(RAW_POI_DATA_PATH)
        print("Finish loading POI data.")

        print("Start processing user information data ... ")
        preprocessed_user_data = data_preprocessing_user_info(preprocessed_trajectory_data)
        PREPROCESSED_USER_DATA_PATH = f"./data/raw/city{city_idx}-user-information-preprocessed.csv"
        store_csv_file(PREPROCESSED_USER_DATA_PATH, preprocessed_user_data)
        print("Finished processing user information data.")

        print("Start processing static graph data ... ")
        preprocessed_static_data = data_preprocessing_static_graph(poi_data, preprocessed_trajectory_data)
        PREPROCESSED_STATIC_DATA_PATH = f"./data/raw/city{city_idx}-static-graph-preprocessed.csv"
        store_csv_file(PREPROCESSED_STATIC_DATA_PATH, preprocessed_static_data)
        print("Finished processing static graph data.")
