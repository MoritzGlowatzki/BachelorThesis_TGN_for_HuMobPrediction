import torch
from tqdm import tqdm

from _1_data_IO import load_csv_file, store_csv_file
from _2b_dataset import UserLocationInteractionDataset
from _3_model_training import TGNModel


# -------- Main Prediction Pipeline -------- #
def run_prediction(prediction_data, city_id):
    # 1) DEVICE CONFIGURATION
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference running on device: {device}")

    # 2) LOAD DATA (and move it onto `device`)
    dataset = UserLocationInteractionDataset(root="data", city_idx=city_id)
    data = dataset[0].to(device)
    print(f"Number of users: {dataset.num_users}")

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
    checkpoint = torch.load("./model_training_runs/best_model.pt", map_location=device)
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
    for idx, (user_id, d, t, _, _) in tqdm(enumerate(prediction_data.itertuples(index=False, name=None)),
                                           total=len(prediction_data)):
        # Step 1: Compute timestamp tensor from day & half-hour slot
        timestamp = int(d * 48 + t)
        timestamp_tensor = torch.tensor([timestamp], dtype=torch.long, device=device)

        # Step 2: Predict top-1 cell ID
        # pred_cell_id, scores = predict_location(user_id, timestamp_tensor, dataset.num_users, device, model, data, top_k=1)
        pred_cell_id, score = predict_next_location(user_id, timestamp_tensor, dataset.num_users, device,
                                                    model, data,
                                                    top_k=5
                                                    )
        pred_cell_id = int(pred_cell_id.item())  # convert single-element tensor to int

        # Step 3: Decode cell_id → (x, y)
        final_results.at[idx, "x"] = (pred_cell_id - 1) % 200 + 1
        final_results.at[idx, "y"] = (pred_cell_id - 1) // 200 + 1

        # Step 4: Prepare message features for memory update
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        loc_tensor = torch.tensor([pred_cell_id], dtype=torch.long, device=device)
        loc_tensor += int(data.src.max())
        user_feats = data.user_feats[user_tensor]
        location_feats = data.location_feats[loc_tensor]

        # Use zeros as dummy edge features (no "real" message at inference)
        edge_feats = torch.zeros((1, data.edge_feats.size(-1)), device=device)

        # Step 5: Update model state
        model.update_states(user_tensor, loc_tensor, timestamp_tensor, user_feats, edge_feats, location_feats)

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
        # raw_prediction_data = raw_prediction_data[raw_prediction_data["uid"] == 3000]
        print(f"There are {len(raw_prediction_data)} predictions to make!")
        final_result = run_prediction(raw_prediction_data, city_idx)
        PREDICTION_RESULT_DATA_PATH = f"./data/result/city{city_idx}_prediction_result.csv"
        store_csv_file(PREDICTION_RESULT_DATA_PATH, final_result)
