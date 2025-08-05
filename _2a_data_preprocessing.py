import numpy as np
from tqdm import tqdm
import argparse, json, os, sys


from _1_data_IO import *


# -------- Data Checks -------- #

def check_for_new_cells_after_day_60(city_idx):
    RAW_FULL_DATA_PATH = f"./data/dataset_humob_2024/full_city_data/city{city_idx}-dataset.csv"
    df = load_csv_file(RAW_FULL_DATA_PATH)

    visited_before = set(map(tuple, df.loc[df["d"] < 60, ["x", "y"]].values))
    visited_after = set(map(tuple, df.loc[df["d"] >= 60, ["x", "y"]].values))
    new_cells_after_60 = visited_after.difference(visited_before)
    total_visited = df.drop_duplicates(subset=["x", "y"]).shape[0]
    if new_cells_after_60:
        print(
            f"{len(new_cells_after_60)} out of {total_visited} cells in city {city_idx} visited after day 60 were new (not seen before).")
    else:
        print(f"No new cells visited in city {city_idx} after day 60 — all were already seen before.")


def total_num_of_records():
    total_num_of_records = 0
    for city_idx in ["A", "B", "C", "D"]:
        RAW_CITY_DATA_PATH = f"./data/dataset_humob_2024/city{city_idx}-groundtruthdata.csv"
        data = load_csv_file(RAW_CITY_DATA_PATH)
        total_num_of_records += len(data)
    print(f"Total Number of Records: {total_num_of_records}")


# -------- Estimations, Calculations, Helper Functions -------- #

def estimate_home_location(df):
    # estimated home = most frequent location on weekends and from 8 PM to 8 AM on weekdays (t in [0–16] and [40–48])
    df = df.copy()
    df["weekday"] = df["d"] % 7
    filtered_df = df[
        (
                ((1 <= df["weekday"]) & (df["weekday"] <= 5)) &  # Weekdays: 1–5
                (
                        ((0 <= df["t"]) & (df["t"] <= 16)) |  # 00:00–08:00
                        ((40 <= df["t"]) & (df["t"] < 48))  # 20:00–24:00
                )
        ) |
        ((df["weekday"] == 0) | (df["weekday"] == 6))  # Weekend: Sunday (0) & Saturday (6)
        ]

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
    # estimated work = most frequent location from 9 AM to 5 PM only on weekdays (t in [18–34])
    df = df.copy()
    df["weekday"] = df["d"] % 7
    filtered_df = df[
        ((1 <= df["weekday"]) & (df["weekday"] <= 5)) &
        ((18 <= df["t"]) & (df["t"] < 34))
        ]

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
    for col in tqdm(df.columns, desc="Downcasting columns"):
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# -------- Data Preprocessing -------- #

def data_preprocessing_trajectory_data(city_data):
    traj_data = city_data.copy()

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

    # cast all columns to int, except:
    # "distance_to_last_position", "distance_from_home", "distance_from_work", "d_sin", "d_cos", "t_sin", "t_cos"
    cols_to_int = traj_data_extended.columns.difference(
        ["distance_to_last_position", "distance_from_home", "distance_from_work", "d_sin", "d_cos", "t_sin", "t_cos"])
    traj_data_extended[cols_to_int] = traj_data_extended[cols_to_int].astype(int)

    return downcast_dataframe(traj_data_extended)


def data_preprocessing_user_data(traj_data):
    user_data = traj_data.copy()

    stats = []

    for uid, group in tqdm(user_data.groupby("uid"), desc="Calculating user stats"):
        home = estimate_home_location(group)
        work = estimate_work_location(group)

        user_locs = list(group["cell_id"].unique())
        user_locs = [int(cell_id) for cell_id in user_locs]

        stats.append({
            "uid": uid,
            "home_x": home["home_x"],
            "home_y": home["home_y"],
            "home_cell_id": home["home_cell_id"],
            "work_x": work["work_x"],
            "work_y": work["work_y"],
            "work_cell_id": work["work_cell_id"],
            "average_movements_per_day": calculate_average_number_of_movements_per_day(group),
            "average_travel_distance_per_day": calculate_average_travel_distance_per_day(group),
            "user_specific_locations": json.dumps(user_locs)
        })

    result = pd.DataFrame(stats)

    # cast all columns except the floats and 'user_specific_locations'
    cols_to_int = result.columns.difference(["average_movements_per_day", "average_travel_distance_per_day", "user_specific_locations"])
    result[cols_to_int] = result[cols_to_int].astype(int)

    return downcast_dataframe(result)


def data_preprocessing_location_data(poi_data, traj_data):
    poi_data = poi_data.copy()
    traj_data = traj_data.copy()

    # calculate cell_id
    poi_data["cell_id"] = poi_data["x"] + (poi_data["y"] - 1) * 200
    traj_data["cell_id"] = traj_data["x"] + (traj_data["y"] - 1) * 200

    # create a dataframe containing all 40000 cells
    static_nodes = pd.DataFrame({
        "cell_id": np.arange(1, 40001),
        "x": np.tile(np.arange(1, 201), 200),  # repeat 1 to 200 200 times for each x
        "y": np.repeat(np.arange(1, 201), 200)  # repeat 1 to 200 for each y
    })

    # POI_feature count per category
    poi_features = (
        poi_data
        .pivot_table(index="cell_id",
                     columns="category",
                     values="POI_count",
                     aggfunc="sum",
                     fill_value=0)
        .reindex(columns=(i for i in range(1, 86)), fill_value=0)  # ensure that all 85 categories are existent
        .reset_index()
        .rename(columns=lambda x: f"POI_cat_{x}" if x != "cell_id" else "cell_id")
    )

    # average dwell time per cell
    avg_dwell_time = (
        traj_data
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
        traj_data
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

    # cast all columns except "avg_dwell_time" to int
    cols_to_int = merged.columns.difference(["visitor_count", "avg_dwell_time"])
    merged[cols_to_int] = merged[cols_to_int].astype(int)

    return downcast_dataframe(merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and add additional features.")
    parser.add_argument("--city", type=str, default="D", help="Comma-separated list of city indices (e.g., A,B,C,D)")
    parser.add_argument("--process", type=str, default="user,trajectory,location", help="Comma-separated processing steps: user, trajectory, location")
    args = parser.parse_args()

    city_list = [city.strip() for city in args.city.split(",")]
    process_list = [p.strip() for p in args.process.split(",")]

    for city_idx in city_list:
        print(f"=== Processing city: {city_idx} ===", flush=True)

        print("Load city data ...", flush=True)
        RAW_CITY_DATA_PATH = f"./data/dataset_humob_2024/city{city_idx}_challengedata.csv"
        city_data = load_csv_file(RAW_CITY_DATA_PATH)
        print("Finished loading city data.", flush=True)

        if city_idx != "A":
            print("Start splitting data ...", flush=True)
            mask = (city_data["x"] == 999) & (city_data["y"] == 999)
            prediction_data = city_data[mask].copy()
            city_data = city_data[~mask]
            RAW_PREDICTION_DATA_PATH = f"./data/raw/city{city_idx}_prediction_data.csv"
            store_csv_file(RAW_PREDICTION_DATA_PATH, prediction_data)
            print("Finished splitting data.", flush=True)

        print("Start processing trajectory data ...", flush=True)
        if "trajectory" in process_list:
            preprocessed_trajectory_data = data_preprocessing_trajectory_data(city_data)
            PREPROCESSED_TRAJECTORY_DATA_PATH = f"./data/raw/city{city_idx}_trajectory_data.csv"
            store_csv_file(PREPROCESSED_TRAJECTORY_DATA_PATH, preprocessed_trajectory_data)
        else:
            PREPROCESSED_TRAJECTORY_DATA_PATH = f"./data/raw/city{city_idx}_trajectory_data.csv"
            if not os.path.exists(PREPROCESSED_TRAJECTORY_DATA_PATH):
                raise FileNotFoundError(f"{PREPROCESSED_TRAJECTORY_DATA_PATH} not found.")
            preprocessed_trajectory_data = load_csv_file(PREPROCESSED_TRAJECTORY_DATA_PATH)
        print("Finished processing trajectory data.", flush=True)

        if "user" in process_list:
            print("Start processing user data ...", flush=True)
            preprocessed_user_data = data_preprocessing_user_data(preprocessed_trajectory_data)
            PREPROCESSED_USER_DATA_PATH = f"./data/raw/city{city_idx}_user_features.csv"
            store_csv_file(PREPROCESSED_USER_DATA_PATH, preprocessed_user_data)
            print("Finished processing user data.", flush=True)

        if "location" in process_list:
            print("Load POI data ...", flush=True)
            RAW_POI_DATA_PATH = f"./data/dataset_humob_2024/POIdata_city{city_idx}.csv"
            poi_data = load_csv_file(RAW_POI_DATA_PATH)
            print("Finish loading POI data.", flush=True)

            print("Start processing location data ...", flush=True)
            preprocessed_location_data = data_preprocessing_location_data(poi_data, preprocessed_trajectory_data)
            PREPROCESSED_LOCATION_DATA_PATH = f"./data/raw/city{city_idx}_location_features.csv"
            store_csv_file(PREPROCESSED_LOCATION_DATA_PATH, preprocessed_location_data)
            print("Finished processing location data.", flush=True)

    print("\n✅ All requested processing complete.")
