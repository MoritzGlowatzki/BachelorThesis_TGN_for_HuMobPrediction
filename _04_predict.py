import pandas as pd
import torch

from _02_b_dataset import UserLocationInteractionDataset
from _03_model_training import TGNModel

# 1) DEVICE CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference running on device: {device}")

# 2) LOAD DATA (and move it onto `device`)
dataset = UserLocationInteractionDataset(root="data", city_idx="D")
data = dataset[0].to(device)
num_users = int(torch.max(data.src).item()) + 1
print(f"Number of users: {num_users}")

# 3) RE-INSTANTIATE MODEL (matching training hyperparameters)
model = TGNModel(
    device=device,
    num_nodes=data.num_nodes,
    msg_dim=data.msg.size(-1),
    memory_dim=100,
    time_dim=100,
    embedding_dim=100,
    neighbor_size=10
)

# 4) LOAD CHECKPOINT (weights only; we’ll rebuild memory & neighbors next)
checkpoint = torch.load("./model_training_runs/best_model.pt", map_location=device)
model.memory.load_state_dict(checkpoint["memory_state"])
model.gnn.load_state_dict(checkpoint["gnn_state"])
model.link_pred.load_state_dict(checkpoint["pred_state"])

# 5) REBUILD FINAL MEMORY & NEIGHBOR STATE FROM HISTORICAL EVENTS
model.eval()
for i in range(data.num_events):
    # Extract the i-th interaction (already on `device`)
    src_i = data.src[i].unsqueeze(0)  # shape [1]
    dst_i = data.dst[i].unsqueeze(0)  # shape [1]
    t_i = data.t[i].unsqueeze(0)  # shape [1]
    msg_i = data.msg[i].unsqueeze(0)  # shape [1, msg_dim]

    # Insert into LastNeighborLoader (keeps a rolling window)
    model.neighbor_loader.insert(src_i, dst_i)

# 6) PREDICT FUNCTION (using memory‐only embeddings, no GNN)
@torch.no_grad()
def predict_next_location(user_id, model, data, top_k=5):
    """
    For a given user_id, return top_k location IDs + their scores, using:
      - user embedding = model.memory(user_id)
      - location embeddings = model.memory(candidate_locations)
    This skips the GNN step at inference to avoid missing‐node issues.
    All tensor ops occur on `device`.
    """
    # (a) get the user's memory embedding
    user_id_tensor = torch.tensor([user_id], dtype=torch.long, device=device)  # [1]
    z_user_mem, _ = model.memory(user_id_tensor)  # [1, memory_dim]

    # (b) build candidate_locations: all location‐node IDs
    #     (assumes users = [0..num_users−1], locations = [num_users..num_nodes−1])
    candidate_locations = torch.arange(
        dataset.num_users,
        data.num_nodes,
        device=device
    )  # [num_locations]

    # (c) get memory embeddings for all candidate locations in one go
    loc_ids = candidate_locations.unsqueeze(1)  # [num_locations, 1]
    loc_ids_flat = loc_ids.view(-1)  # [num_locations]
    z_loc_mem, _ = model.memory(loc_ids_flat)  # [num_locations, memory_dim]

    # (d) score user → each location via link_pred
    num_loc = z_loc_mem.size(0)
    user_emb_batch = z_user_mem.expand(num_loc, -1)  # [num_locations, memory_dim]
    scores = model.link_pred(user_emb_batch, z_loc_mem).squeeze(1)  # [num_locations]

    # (e) pick top_k
    top_scores, top_indices = torch.topk(scores, k=top_k)
    top_locations = candidate_locations[top_indices]  # [top_k], on device

    return top_locations, top_scores  # both CUDA tensors if device=="cuda"


# 7) AUTOREGRESSIVE INFERENCE LOOP (simulate next 720 timestamps)
t0 = data.t.max().item()
future_predictions = []  # switch to a list of dicts for (user_id, timestamp) pairs

for dt in range(1, 2):  # 721
    pred_time = t0 + dt

    # ─── Loop over every user ───
    for user_id in range(num_users):
        # 7.1) Predict next location(s) for this user_id at time = t0 + dt
        locs, scores = predict_next_location(user_id, model, data, top_k=5)
        top_loc = locs[0]  # scalar tensor on device
        top_score = scores[0]  # scalar tensor on device

        # 7.2) Record them (move to CPU for storage)
        future_predictions.append({
            "user_id": user_id,
            "timestamp": pred_time,
            "locations": locs.cpu().tolist(),
            "scores": scores.cpu().tolist()
        })

        # 7.3) Feed top‐1 back into TGN so memory & neighbor history update
        pred_user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)  # [1]
        pred_loc_tensor = top_loc.unsqueeze(0)  # [1], already on device
        pred_time_tensor = torch.tensor([pred_time], dtype=torch.long, device=device)  # [1]

        # If no “real” message feature exists for a future check‐in, use a zero‐vector
        fake_msg = torch.zeros((1, data.msg.size(-1)), device=device)  # [1, msg_dim]

        model.update_states(pred_user_tensor, pred_loc_tensor, pred_time_tensor, fake_msg)

# 8) SAVE TOP‐PREDICTION PER (user, timestamp) INTO CSV
rows = []
for pred in future_predictions:
    rows.append({
        "user_id": pred["user_id"],
        "timestamp": pred["timestamp"],
        "predicted_location": pred["locations"][0],
        "score": pred["scores"][0]
    })

df = pd.DataFrame(rows)
df.to_csv("./prediction/top_predictions.csv", index=False)
print("Top predictions saved!")
print(df.head(10))
