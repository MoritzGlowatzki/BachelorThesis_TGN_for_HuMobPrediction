import argparse
import ast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

from _1_data_IO import load_csv_file, store_csv_file
from _2b_dataset import UserLocationInteractionDataset
from _3_model_training import TGNModel


# -------- Dataset for Prediction batched by Timestamp -------- #
class TimestampBatchedPredictionDataset(Dataset):
    def __init__(self, city_idx, num_users_during_training):
        self.city_idx = city_idx
        self.num_users_during_training = num_users_during_training
        self.pred_data = load_csv_file(f"./data/raw/city{self.city_idx}_prediction_data.csv")
        self.user_data = load_csv_file(f"./data/raw/city{self.city_idx}_user_features.csv")
        # self.location_data = load_csv_file(f"./data/raw/city{self.city_idx}_location_features.csv")

        # Compute timestamp
        self.pred_data["timestamp"] = (self.pred_data["d"] * 48 + self.pred_data["t"]).astype(int)

        # Load candidate locations and parse lists
        candidates_df = self.user_data.copy()
        candidates_df["candidate_locations"] = candidates_df["user_specific_locations"].apply(ast.literal_eval)
        candidates_df = candidates_df[["uid", "candidate_locations"]]

        # Merge into main data
        self.pred_data = self.pred_data.merge(candidates_df, on="uid", how="left")

        # Group indices by timestamp
        self.timestamp_to_indices = {}
        for idx, ts in enumerate(self.pred_data["timestamp"]):
            if ts not in self.timestamp_to_indices:
                self.timestamp_to_indices[ts] = []
            self.timestamp_to_indices[ts].append(idx)

        # Sorted list of timestamps
        self.timestamps = sorted(self.timestamp_to_indices.keys())

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        ts = ts[0]
        indices = self.timestamp_to_indices[ts]
        batch_df = self.pred_data.iloc[indices]

        # Build flattened edge list
        src_list, dst_list = [], []
        for uid, cand_list in zip(batch_df["uid"], batch_df["candidate_locations"]):
            src_list.extend([uid] * len(cand_list))
            dst_list.extend(cand_list)

        src = torch.tensor(src_list, dtype=torch.long)
        dst = torch.tensor(dst_list, dtype=torch.long)
        dst += self.num_users_during_training  # offset candidate location IDs to match the training ID space
        t = torch.full((len(src),), ts, dtype=torch.long)

        return TemporalData(
            src=src,
            dst=dst,
            t=t,
        )


# -------- Helper Function -------- #

def find_user_feats(user_data, uids):
    # filter the rows for all uids
    user_infos = user_data[user_data["uid"].isin(uids)].drop(columns=["user_specific_locations"])

    # ensure the order matches uids
    user_infos = user_infos.set_index("uid").loc[uids].reset_index()

    # convert to torch tensor
    user_feats = torch.tensor(user_infos.values, dtype=torch.float)
    return user_feats


def calculate_edge_feats(new_src, new_dst, new_t, user_data, last_records):
    # Copy last_records to avoid modifying original
    new_records = last_records.copy()

    # Convert inputs to Series for vectorized operations
    new_src = pd.Series(new_src, name="uid")
    new_dst = pd.Series(new_dst, name="cell_id")
    new_t = pd.Series(new_t, name="timestamp")

    # Get previous user records
    prev_records = new_records.set_index("uid").loc[new_src].reset_index()

    # Extract previous positions and time deltas
    x_prev = prev_records["x"].values
    y_prev = prev_records["y"].values
    timestamp_prev = prev_records["timestamp"].values
    timedelta_prev = prev_records["timedelta_since_last_movement"].values

    # Get user-specific info
    user_info = user_data.set_index("uid").loc[new_src].reset_index()
    home_x = user_info["home_x"].values
    home_y = user_info["home_y"].values
    work_x = user_info["work_x"].values
    work_y = user_info["work_y"].values

    # Compute new positions
    x = (new_dst - 1) % 200 + 1
    y = (new_dst - 1) // 200 + 1

    # Check if user moved
    is_recorded = (x != x_prev) | (y != y_prev)
    is_recorded = is_recorded.astype(int)

    # Compute day and time features
    d = new_t // 48
    t = new_t % 48

    d_sin = np.sin(2 * np.pi * d / 7)
    d_cos = np.cos(2 * np.pi * d / 7)
    t_sin = np.sin(2 * np.pi * t / 48)
    t_cos = np.cos(2 * np.pi * t / 48)

    # Weekday dummies
    weekday = d % 7
    wd_dummies = pd.get_dummies(weekday, prefix="wd").reindex(columns=[f"wd_{i}" for i in range(7)], fill_value=0)

    weekend = ((weekday == 0) | (weekday == 6)).astype(int)
    daytime = ((14 <= t) & (t < 38)).astype(int)

    # Time delta since last movement
    same_position = (x == x_prev) & (y == y_prev)
    timedelta_since_last_movement = np.where(
        same_position,
        new_t - timestamp_prev + timedelta_prev,
        0
    )

    # Distances
    distance_from_home = np.sqrt((x - home_x) ** 2 + (y - home_y) ** 2)
    distance_from_work = np.sqrt((x - work_x) ** 2 + (y - work_y) ** 2)
    distance_to_last_position = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

    # Position flags
    position_flag = np.where(
        (x == home_x) & (y == home_y), 0,
        np.where((x == work_x) & (y == work_y), 1, 2)
    )

    # Prepare dataframe to update last_records
    update_df = pd.DataFrame({
        "uid": new_src,
        "d": d, "t": t, "x": x, "y": y, "is_recorded": is_recorded,
        "cell_id": new_dst, "timestamp": new_t,
        "d_sin": d_sin, "d_cos": d_cos, "t_sin": t_sin, "t_cos": t_cos,
        "weekend": weekend, "daytime": daytime,
        "timedelta_since_last_movement": timedelta_since_last_movement,
        "distance_from_home": distance_from_home,
        "distance_from_work": distance_from_work,
        "distance_to_last_position": distance_to_last_position,
        "position_flag": position_flag
    })

    # Add weekday dummies
    update_df = pd.concat([update_df, wd_dummies.reset_index(drop=True)], axis=1)

    # Convert integer columns
    cols_to_int = update_df.columns.difference(
        ["distance_to_last_position", "distance_from_home", "distance_from_work", "d_sin", "d_cos", "t_sin", "t_cos"]
    )
    update_df[cols_to_int] = update_df[cols_to_int].astype(int)

    # Update new_records
    new_records = new_records.set_index("uid")
    update_df = update_df.set_index("uid")
    new_records.update(update_df)
    new_records = new_records.reset_index()

    # Extract edge features
    edge_infos = new_records[new_records["uid"].isin(new_src)]
    edge_feats = torch.tensor(edge_infos.drop(columns=["uid", "d", "t", "x", "y", "cell_id"]).values, dtype=torch.float)

    return edge_feats, new_records


def find_location_feats(location_data, cell_ids):
    # filter the rows for all cell_ids
    location_infos = location_data[location_data["cell_id"].isin(cell_ids)]

    # ensure the order matches cell_ids
    location_infos = location_infos.set_index("cell_id").loc[cell_ids].reset_index()

    # convert to torch tensor
    location_feats = torch.tensor(location_infos.values, dtype=torch.float)
    return location_feats


# -------- Main Inference Pipeline -------- #
def run_inference(raw_prediction_data, city_idx, small, interpol, model_name, feats):
    # 1) DEVICE CONFIGURATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference running on device: {device}", flush=True)

    # 2) LOAD DATA (and move it onto device)
    dataset = UserLocationInteractionDataset(root="data", city_idx=city_idx, small=small, interpol=interpol, feats=feats)
    data = dataset[0].to(device)
    print(f"Number of users: {dataset.num_users}", flush=True)
    print(f"Number of visited locations: {dataset.num_visited_locations}", flush=True)

    # 3) RE-INSTANTIATE MODEL (matching training hyperparameters)
    model = TGNModel(
        device=device,
        num_nodes=data.num_nodes,
        msg_dim=(data.user_feats.size(-1) + data.edge_feats.size(-1) + data.location_feats.size(-1)),
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        neighbor_size=10,
    )

    # 4) LOAD CHECKPOINT (weights and final memory & neighbor state)
    checkpoint = torch.load(f"./model_training_runs/{model_name}.pt", map_location=device)
    model.memory.load_state_dict(checkpoint["memory_state"])
    model.gnn.load_state_dict(checkpoint["gnn_state"])
    model.link_pred.load_state_dict(checkpoint["pred_state"])
    model.memory.memory = checkpoint["memory_buffer"].to(device)
    model.memory.last_update = checkpoint["last_update"].to(device)
    model.neighbor_loader.neighbors = checkpoint["neighbor_dict"].to(device)
    # model.assoc = checkpoint["assoc"].to(device)

    if "epoch" in checkpoint:
       print(f"Loaded model from epoch {checkpoint['epoch']}", flush=True)

    # 5) PUT MODEL INTO EVALUATION MODE
    model.memory.eval()
    model.gnn.eval()
    model.link_pred.eval()

    # 6) Create DataLoader for BatchedPredictionDataset
    num_users_during_training = int(data.src.max())
    pred_data = TimestampBatchedPredictionDataset(city_idx=city_idx,
                                                  num_users_during_training=num_users_during_training)
    pred_loader = TemporalDataLoader(pred_data, batch_size=1, shuffle=False)

    # 7) Load auxiliary information
    print("Start loading auxiliary data ...", flush=True)
    user_data = load_csv_file(f"./data/raw/city{city_idx}_user_features.csv")
    location_data = load_csv_file(f"./data/raw/city{city_idx}_location_features.csv")
    traj_data = load_csv_file(f"./data/raw/city{city_idx}_trajectory_data.csv")
    print("Finished loading auxiliary data successfully!", flush=True)

    idx = traj_data.groupby("uid")["timestamp"].idxmax()
    last_records = traj_data.loc[idx]

    # 8) AUTOREGRESSIVE INFERENCE LOOP
    num_users_during_training = int(data.src.max())
    return batch_predict_next_locations(model, pred_loader, data, raw_prediction_data, user_data, location_data,
                                        last_records, num_users_during_training)


@torch.no_grad()
def batch_predict_next_locations(model, loader, data, prediction_data, user_data, location_data, last_records,
                                 num_users_during_training):
    model.memory.eval()
    model.gnn.eval()
    model.link_pred.eval()

    final_results = prediction_data.copy()
    final_results.set_index(["uid", "d", "t"], inplace=True, drop=False)

    # TRUE_DATA_PATH = "./data/dataset_humob_2024/full_city_data/cityD-dataset.csv"
    # ground_truth = load_csv_file(TRUE_DATA_PATH)
    # ground_truth["cell_id"] = (ground_truth["x"] + (ground_truth["y"] - 1) * 200).astype(int)
    # aps, aucs = [], []

    for batch in tqdm(loader, desc="Processing batches"):
        batch = batch.to(model.device)

        z = model.compute_embeddings(batch, data)

        # collect predicted edges
        all_src = []
        all_dst = []
        all_t = []
        for uid in batch.src.unique():
            user_mask = batch.src == uid  # find indices in src that belong to this user by creating a mask

            uid_tensor = batch.src[user_mask]  # get all user nodes
            candidate_location_tensor = batch.dst[user_mask]  # get all candidate locations for user
            # candidate_location_tensor += num_users_during_training
            timestamp = batch.t.unique()

            scores = model.predict_inference_scores_for_single_user(z, uid_tensor, candidate_location_tensor, timestamp)

            # find the index and therefore the candidate location that is the most likely one
            max_idx = torch.argmax(scores)
            predicted_cell = candidate_location_tensor[
                                 max_idx] - num_users_during_training  # revert offset to match the training ID space
            # print(f"User {uid} is predicted to be in cell {predicted_cell} "
            #       f"at time {timestamp} with a probability of {scores[max_idx].sigmoid().item():.4f}")

            # keep only the edge to the predicted cell for later updating the model states
            all_src.append(uid_tensor[max_idx].item())
            all_dst.append(predicted_cell.item())
            all_t.append(timestamp.item())

            # decode cell_id → (x, y)
            x = (predicted_cell.cpu().item() - 1) % 200 + 1
            y = (predicted_cell.cpu().item() - 1) // 200 + 1

            # decode timestamp -> (d, t)
            d = timestamp.cpu().item() // 48
            t = timestamp.cpu().item() % 48

            # copy the tensor to CPU memory
            uid = uid.cpu().item()

            # # Determine if ground truth is in candidate set
            # true_location = ground_truth.loc[(ground_truth["uid"] == uid) & (ground_truth["d"] == d) & (ground_truth["t"] == t), "cell_id"].iloc[0]
            # candidate_locations = candidate_location_tensor.cpu().tolist()
            # if true_location in candidate_locations:
            #     # Positive index
            #     pos_out = scores[candidate_locations == true_location]
            #     # Negatives are all other candidates
            #     neg_out = scores[candidate_locations != true_location]
            #     y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            #     y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)

            #     aps.append(average_precision_score(y_true, y_pred))
            #     aucs.append(roc_auc_score(y_true, y_pred))
            # else:
            #     # Ground truth not in candidates => model cannot predict, assign 0
            #     aps.append(0.0)
            #     aucs.append(0.0)

            # write predictions in the final dataframe (uid, d, t, x, y)
            final_results.at[(uid, d, t), "x"] = x
            final_results.at[(uid, d, t), "y"] = y

        # prepare features of the entire batch
        user_feats = find_user_feats(user_data, all_src)
        edge_feats, last_records = calculate_edge_feats(all_src, all_dst, all_t, user_data, last_records)
        location_feats = find_location_feats(location_data, all_dst)

        # create batch for update
        batch_for_update = TemporalData(
            src=torch.tensor(all_src, dtype=torch.long, device=model.device),
            dst=torch.tensor(all_dst, dtype=torch.long, device=model.device),
            t=torch.tensor(all_t, dtype=torch.long, device=model.device),
            user_feats=user_feats.to(model.device),
            edge_feats=edge_feats.to(model.device),
            location_feats=location_feats.to(model.device)
        )

        # update the models states and all with the predicted user->location interaction
        model.update_states(batch_for_update.src, batch_for_update.dst, batch_for_update.t,
                            batch_for_update.user_feats, batch_for_update.edge_feats, batch_for_update.location_feats)


    # mean_ap = float(torch.tensor(aps).mean())
    # mean_auc = float(torch.tensor(aucs).mean())

    # print(f"Mean Average Precision (AP): {mean_ap:.4f}")
    # print(f"Mean Area Under Curve (AUC): {mean_auc:.4f}")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and add additional features.")
    parser.add_argument("--city", type=str, default="D", help="City index (e.g., A, B, C, D)")
    parser.add_argument("--small", type=bool, default=False, help="Only use users that will be predicted later")
    parser.add_argument("--interpol", type=bool, default=True, help="Interpolation Yes/No")
    parser.add_argument("--feats", type=bool, default=True, help="Include features Yes/No")
    parser.add_argument("--model", type=str, default="last_model_D_50", help="Used model")
    args = parser.parse_args()


    RAW_PREDICTION_DATA_PATH = f"./data/raw/city{args.city}_prediction_data.csv"
    raw_prediction_data = load_csv_file(RAW_PREDICTION_DATA_PATH)

    print(f"There are {len(raw_prediction_data)} predictions to make!")
    final_result = run_inference(raw_prediction_data, args.city, args.small, args.interpol, args.model, args.feats)
    
    PREDICTION_RESULT_DATA_PATH = f"./data/result/city{args.city}_prediction_result_{args.model}.csv"
    store_csv_file(PREDICTION_RESULT_DATA_PATH, final_result)
