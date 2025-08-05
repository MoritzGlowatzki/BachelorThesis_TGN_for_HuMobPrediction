import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from _1_data_IO import load_csv_file, store_csv_file
from _2a_data_preprocessing import downcast_dataframe, calculate_euclidean_distance
from _2b_dataset import UserLocationInteractionDataset
from _3_model_training import TGNModel


# -------- Main Prediction Pipeline -------- #
def cell_id_to_coordinate(cell_id):
    x = (cell_id - 1) % 200 + 1
    y = (cell_id - 1) // 200 + 1
    return x, y


def compute_edge_infos(user_id, d, t, x, y, home, work, time_since_last_movement, x_prev, y_prev):
    edge_infos = []
    record = pd.DataFrame({
        "uid": user_id,
        "d": d,
        "t": t,
        "x": x,
        "y": y
    })
    record["is_recorded"] = 0
    record["cell_id"] = ((record["x"] + (record["y"] - 1) * 200).astype(int))
    record["timestamp"] = (record["d"] * 48 + record["t"]).astype(int)
    record["d_sin"] = np.sin(2 * np.pi * record["d"] / 7)
    record["d_cos"] = np.cos(2 * np.pi * record["d"] / 7)
    record["t_sin"] = np.sin(2 * np.pi * record["t"] / 48)
    record["t_cos"] = np.cos(2 * np.pi * record["t"] / 48)
    record["weekday"] = record["d"] % 7
    record = pd.get_dummies(record, columns=["weekday"], prefix="wd")
    record["weekend"] = (record["wd_0"] == 1) | (record["wd_6"] == 1)
    record["daytime"] = ((14 <= record["t"]) & (record["t"] < 38))
    record["timedelta_since_last_movement"] = time_since_last_movement + 1 if x == x_prev and y == y_prev else 0
    home_x, home_y = cell_id_to_coordinate(home)
    record["distance_from_home"] = calculate_euclidean_distance(x, y, home_x, home_y)
    work_x, work_y = cell_id_to_coordinate(work)
    record["distance_from_work"] = calculate_euclidean_distance(x, y, work_x, work_y)
    record["distance_to_last_position"] = calculate_euclidean_distance(x, y, x_prev, y_prev)
    record["position_flag"] = 2
    record.loc[(x == home_x) & (y == home_y), "position_flag"] = 0
    record.loc[(x == work_x) & (y == work_y), "position_flag"] = 1
    cols_to_int = record.columns.difference(
        ["distance_to_last_position", "distance_from_home", "distance_from_work", "d_sin", "d_cos", "t_sin", "t_cos"])
    record[cols_to_int] = record[cols_to_int].astype(int)

    return downcast_dataframe(record)

    return edge_infos


def run_prediction(prediction_data, city_id):
    # 1) DEVICE CONFIGURATION
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference running on device: {device}")

    # 2) LOAD DATA (and move it onto device)
    dataset = UserLocationInteractionDataset(root="data", city_idx=city_id)
    data = dataset[0].to(device)
    print(f"Number of users: {data.num_users}")
    print(f"Number of nodes: {data.num_nodes}")

    # 3) RE-INSTANTIATE MODEL (matching training hyperparameters)
    model = TGNModel(
        num_nodes=data.num_nodes,
        msg_dim=(data.user_feats.size(-1) + data.edge_feats.size(-1) + data.location_feats.size(-1)),
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        neighbor_size=10,
        device=device
    )

    # 4) LOAD CHECKPOINT (weights and final memory & neighbor state)
    checkpoint = torch.load("./model_training_runs/best_model_D_small_no_interpolation.pt", map_location=device)
    model.memory.load_state_dict(checkpoint["memory_state"])
    model.gnn.load_state_dict(checkpoint["gnn_state"])
    model.link_pred.load_state_dict(checkpoint["pred_state"])
    model.memory.memory = checkpoint["memory_buffer"].to(device)
    model.memory.last_update = checkpoint["last_update"].to(device)
    model.neighbor_loader.neighbors = checkpoint["neighbor_dict"].to(device)

    # 5) PUT MODEL INTO EVALUATION MODE
    model.eval()

    # 6) AUTOREGRESSIVE INFERENCE LOOP
    final_results = prediction_data.copy()
    time_since_last_movement = 0
    x_prev = 0
    y_prev = 0
    for idx, (user_id, d, t, _, _) in tqdm(enumerate(prediction_data.itertuples(index=False, name=None)),
                                           total=len(prediction_data)):
        # Step 1: Compute timestamp tensor from day & half-hour slot
        timestamp = int(d * 48 + t)
        timestamp_tensor = torch.tensor([timestamp], dtype=torch.long, device=device)

        # Step 2: Predict top-1 cell ID
        # pred_cell_id, scores = predict_location(user_id, timestamp_tensor, dataset.num_users, device, model, data, top_k=1)
        pred_cell_id, score = predict_next_location(user_id, timestamp_tensor, data.num_users, device, model, data,
                                                    top_k=5)
        pred_cell_id = int(pred_cell_id.item())  # convert single-element tensor to int

        # Step 3: Decode cell_id → (x, y)
        x = (pred_cell_id - 1) % 200 + 1
        y = (pred_cell_id - 1) // 200 + 1
        final_results.at[idx, "x"] = x
        final_results.at[idx, "y"] = y

        # Step 4: Prepare message features for memory update
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        loc_tensor = torch.tensor([pred_cell_id], dtype=torch.long, device=device)
        loc_tensor += int(data.src.max())
        user_feats = data.user_feats[user_tensor]
        location_feats = data.location_feats[loc_tensor]

        record = compute_edge_infos(user_id, d, t, x, y, user_feats["home"], user_feats["work"],
                                    time_since_last_movement, x_prev, y_prev)
        edge_infos = record.drop(columns=["uid", "d", "t", "x", "y", "cell_id"]).values.tolist()
        edge_feats = torch.tensor(edge_infos, dtype=torch.float)

        # Step 5: Update model state
        model.update_states(user_tensor, loc_tensor, timestamp_tensor, user_feats, edge_feats, location_feats)

        # Step 6: Update variables
        time_since_last_movement = record["time_since_last_movement"]
        x_prev = x, y_prev = y

    return final_results

@torch.no_grad()
def predict_next_location(user_id, t, num_users, device, model, data, top_k=5):
    """
    GNN-powered inference:
      1) Build a “batch” consisting of the single user node plus ALL candidate location nodes.
      2) Run model.compute_embeddings(...) to get TGN memory+GNN embeddings for those nodes.
      3) Score user→each location with link_pred (including the time feature).
      4) Return top_k location IDs and their scores.
    """
    # 1) Prepare node list: [user, loc0, loc1, ..., locN]
    user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
    # location nodes in the graph are indexed num_users ... num_nodes-1
    candidate_locations = torch.arange(num_users,
                                       data.num_nodes,
                                       dtype=torch.long,
                                       device=device)

    n_id = torch.cat([user_tensor, candidate_locations], dim=0)

    # 2) Create a minimal “batch‐like” object for compute_embeddings()
    class _Batch:
        pass

    batch = _Batch()
    batch.n_id = n_id

    # 3) Compute TGN embeddings (memory + GNN) for all these nodes
    z = model.compute_embeddings(batch, data)
    # z[0] is the user embedding, z[1:] are the candidate location embeddings

    # 4) Score user→each location
    z_dst = z[1:]  # candidate locations
    z_src = z[0].unsqueeze(0).repeat(z_dst.size(0), 1)  # [num_locations, dim]
    scores = model.link_pred(z_src, z_dst, t).squeeze(-1)  # [num_locations]

    # 5) Pick top_k
    top_scores, top_idx = torch.topk(scores, k=top_k)
    top_locations = candidate_locations[top_idx] - num_users  # map back to “cell IDs”

    # Print matching cell IDs and scores
    # print("Top 5 Predictions (cell_id, score):")
    # for i in range(top_k):
    #     loc_id = int(top_locations[i].item())
    #     score = float(top_scores[i].item())
    #     print(f"  Cell ID: {loc_id}, Score: {score}")

    return top_locations[0], top_scores[0]


if __name__ == '__main__':
    for city_idx in ["D"]:
        RAW_PREDICTION_DATA_PATH = f"./data/raw/city{city_idx}_prediction_data.csv"
        raw_prediction_data = load_csv_file(RAW_PREDICTION_DATA_PATH)

        # preliminary
        raw_prediction_data = raw_prediction_data[raw_prediction_data["uid"] == 3000]

        print(f"There are {len(raw_prediction_data)} predictions to make!")
        final_result = run_prediction(raw_prediction_data, city_idx)
        PREDICTION_RESULT_DATA_PATH = f"./data/result/city{city_idx}_prediction_result.csv"
        store_csv_file(PREDICTION_RESULT_DATA_PATH, final_result)
