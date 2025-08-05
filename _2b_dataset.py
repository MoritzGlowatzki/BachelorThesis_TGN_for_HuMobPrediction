import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, TemporalData
import argparse, json, sys


class UserLocationInteractionDataset(InMemoryDataset):
    def __init__(self, root, city_idx, transform=None, pre_transform=None):
        self.city_idx = city_idx
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # relative to root/raw/
        return [f"city{self.city_idx}_trajectory_data.csv",
                f"city{self.city_idx}_user_features.csv",
                f"city{self.city_idx}_location_features.csv"]

    @property
    def processed_file_names(self):
        # will be saved under root/processed/
        return [f"interaction-graph-city{self.city_idx}-small-no-interpolation.pt"]

    def process(self) -> None:
        # load the CSV into a Pandas DataFrame.
        traj_data = pd.read_csv(self.raw_paths[0])
        user_data = pd.read_csv(self.raw_paths[1])
        location_data = pd.read_csv(self.raw_paths[2])

        # exclude interpolated observation
        df = traj_data[traj_data["is_recorded"] == 1]

        max_uid = df["uid"].max()
        print(f"Max uid in filtered DataFrame: {max_uid}")

        # TODO: DELETE LATER
        # do not use the entire dataset, but the users to predict in cityD
        df = df[(df["uid"] >= 3000) & (df["uid"] <= 6000)]

        num_users = df["uid"].nunique()
        num_visited_locations = df["cell_id"].nunique()

        src = []  # user nodes
        dst = []  # location nodes
        timestamps = []
        user_infos = []
        location_infos = []
        edge_infos = []

        for uid, group in df.groupby("uid"):
            # create edges: from user to each location (uid → cell_id)
            locs = group["cell_id"].tolist()
            src.extend([uid] * len(locs))
            dst.extend(locs)
            timestamps.extend(group["timestamp"].tolist())

            user_info = user_data[user_data["uid"] == uid].drop(columns=["user_specific_locations"]).values.tolist()
            user_infos.extend(user_info * len(group))  # replicate user feature per edge

            location_info = (location_data[location_data["cell_id"].isin(group["cell_id"])]
                             .set_index("cell_id")
                             .loc[group["cell_id"]]
                             .reset_index()
                             .values.tolist())
            location_infos.extend(location_info)

            edge_infos.extend(
                group.drop(columns=["uid", "d", "t", "x", "y", "cell_id"]).values.tolist())

        # convert to tensors
        src = torch.tensor(src, dtype=torch.long)  # [E]
        dst = torch.tensor(dst, dtype=torch.long)  # [E]
        dst += int(src.max())  # location cell_ids start after user ids
        timestamps = torch.tensor(timestamps, dtype=torch.long)  # [E]
        user_feats = torch.tensor(user_infos, dtype=torch.float)  # [E, num_user _features]
        location_feats = torch.tensor(location_infos, dtype=torch.float)  # [E, num_location_features]
        edge_feats = torch.tensor(edge_infos, dtype=torch.float)  # [E, num_edge_features]

        # globally sort by timestamp t_all to ensure strict time causality
        sorted_idx = torch.argsort(timestamps)
        src = src[sorted_idx]
        dst = dst[sorted_idx]
        timestamps = timestamps[sorted_idx]
        edge_feats = edge_feats[sorted_idx]

        # build a TemporalData object
        data = TemporalData(
            src=src,  # [E]
            dst=dst,  # [E]
            t=timestamps,  # [E]
            user_feats=user_feats,  # [E, num_user _features]
            location_feats=location_feats,  # [E, num_location_features]
            edge_feats=edge_feats,  # [E, num_edge_features]
        )

        data.num_users = num_users
        data.num_visited_locations = num_visited_locations

        torch.save((self.collate([data])), self.processed_paths[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data and add additional features.")
    parser.add_argument("--city", type=str, default="D", help="City index (e.g., A, B, C, D)")
    args = parser.parse_args()

    print("=== User-Location Interaction Dataset ===", flush=True)
    dataset = UserLocationInteractionDataset(root="data", city_idx=args.city)
    data = dataset[0]

    print("\n=== Sanity Checks ===")
    print(f"Total number of edges: {data.src.size(0)}")
    print(f"Unique users (src): {torch.unique(data.src).numel()}")
    print(f"Unique locations (dst): {torch.unique(data.dst).numel()}")
    print(f"Timestamps shape: {data.t.shape}")
    print(f"User node features shape: {data.user_feats.shape}")
    print(f"Edge features shape: {data.edge_feats.shape}")
    print(f"Location node features shape: {data.location_feats.shape}")

    # check timestamp ordering
    timestamps = data.t.flatten()
    timestamp_diffs = timestamps[1:] - timestamps[:-1]
    assert torch.all(timestamp_diffs >= 0).item(), "Timestamps are not sorted!"
    print("Timestamps are sorted.")

    # check proper ID offset
    offset = int(data.src.max()) + 1
    min_dst = int(data.dst.min())
    assert min_dst >= offset, f"Location IDs not offset correctly: {min_dst} < {offset}"
    print(f"Location node offset verified (min dst = {min_dst}, offset = {offset})")

    # print a random edge sample
    import random

    idx = random.randint(0, data.src.size(0) - 1)
    print(f"\n=== Random Edge [{idx}] ===", flush=True)
    print(f"User ID (src): {data.src[idx].item()}")
    print(f"Location ID (dst): {data.dst[idx].item()}")
    print(f"Timestamp: {data.t[idx].item()}")
    print(f"User features: {data.user_feats[idx].tolist()}")
    print(f"Edge features: {data.edge_feats[idx].tolist()}")
    print(f"Location features: {data.location_feats[idx].tolist()}")
    print(f"Message: {torch.cat([data.user_feats[idx], data.edge_feats[idx], data.location_feats[idx]], dim=-1)}")

    # Parse JSON strings in the column "user_specific_locations"
    user_specific_locations = pd.read_csv(
        "./data/raw/cityD_user_features.csv",
        usecols=["uid", "user_specific_locations"]
    )
    user_specific_locations["user_specific_locations"] = user_specific_locations["user_specific_locations"].apply(json.loads)
    locations = user_specific_locations.loc[user_specific_locations["uid"] == data.src[idx].item(), "user_specific_locations"]
    print(f"User specific locations for user {data.src[idx].item()}: {locations.values[0]}")

    print("\n✅ All checks complete.")
