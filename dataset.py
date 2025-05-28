import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, TemporalData

from I_data_IO import load_csv_file


class TrajectoryDataset(InMemoryDataset):
    def __init__(self, root, city_idx, transform=None, pre_transform=None):
        self.city_idx = city_idx
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # relative to root/raw/
        return [f"city{self.city_idx}-trajectory-dataset-preprocessed.csv"]

    @property
    def processed_file_names(self):
        # will be saved under root/processed/
        return [f"trajectory-graph-city{self.city_idx}.pt"]

    def process(self) -> None:
        # load the CSV into a Pandas DataFrame.
        raw_path = self.raw_paths[0]
        df = pd.read_csv(raw_path)

        # Build src/dst “pairs” by shifting the cell_id sequence by one
        src_cell_ids = df["cell_id"].iloc[:-1].to_numpy(dtype=int)  # L₁, L₂, …
        dst_cell_ids = df["cell_id"].iloc[1:].to_numpy(dtype=int)  # L₂, L₃, …
        timestamps = df["timestamp"].iloc[1:].to_numpy(
            dtype=int)  # time of L₂ (event that user arrives at “destination”)
        user_ids = df["uid"].iloc[1:].to_numpy(dtype=int)
        is_recorded_flags = df["is_recorded"].iloc[1:].to_numpy(dtype=int)

        # Convert to tensor
        src_cell_ids = torch.tensor(src_cell_ids, dtype=torch.long)  # [E]
        dst_cell_ids = torch.tensor(dst_cell_ids, dtype=torch.long)  # [E]
        timestamps = torch.tensor(timestamps, dtype=torch.float)  # [E], floats for time‐encoder
        edge_features = torch.tensor(
            np.stack([user_ids, is_recorded_flags], axis=1),  # we store uid and is_recorded as the edge attribute
            dtype=torch.long
        )  # [E,2]

        # Globally sort by timestamp t_all to ensure strict time causality
        sorted_idx = torch.argsort(timestamps)
        src_cell_ids = src_cell_ids[sorted_idx]
        dst_cell_ids = dst_cell_ids[sorted_idx]
        timestamps = timestamps[sorted_idx]
        edge_features = edge_features[sorted_idx]

        # Build a TemporalData object
        data = TemporalData(
            src=src_cell_ids,  # [E]
            dst=dst_cell_ids,  # [E]
            t=timestamps,  # [E]
            edge_attr=edge_features,  # [E, num_edge_features]
        )
        torch.save(self.collate([data]), self.processed_paths[0])


class UserLocationInteractionDataset(InMemoryDataset):
    def __init__(self, root, city_idx, transform=None, pre_transform=None):
        self.city_idx = city_idx
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # relative to root/raw/
        return [f"city{self.city_idx}-trajectory-dataset-preprocessed.csv"]

    @property
    def processed_file_names(self):
        # will be saved under root/processed/
        return [f"interaction-graph-city{self.city_idx}-small.pt"]

    def process(self) -> None:
        # load the CSV into a Pandas DataFrame.
        raw_path = self.raw_paths[0]
        df = pd.read_csv(raw_path)
        df = df[df["uid"] == 0]

        # Build src/dst “pairs” by connecting each stay to the user node
        src = []
        dst = []
        timestamps = []
        is_recorded = []

        # Group by user
        for uid, group in df.groupby("uid"):
            # Create edges: from user to each location
            src.extend([uid] * len(group))  # uid → cell_id
            dst.extend(group["cell_id"].tolist())
            timestamps.extend(group["timestamp"].tolist())
            is_recorded.extend(group["is_recorded"].tolist())

        # Convert to tensors
        src = torch.tensor(src, dtype=torch.long)  # [E]
        dst = torch.tensor(dst, dtype=torch.long)  # [E]
        dst += int(src.max()) + 1
        timestamps = torch.tensor(timestamps, dtype=torch.long)  # [E]
        edge_attr = torch.tensor(is_recorded, dtype=torch.float).unsqueeze(
            1)  # [E,1] -> unsqueeze because edge_attr is expected to be a 2-D tensor of shape [num_edges, num_edge_features]

        # Globally sort by timestamp t_all to ensure strict time causality
        sorted_idx = torch.argsort(timestamps)
        src = src[sorted_idx]
        dst = dst[sorted_idx]
        timestamps = timestamps[sorted_idx]
        edge_attr = edge_attr[sorted_idx]

        # Build a TemporalData object
        data = TemporalData(
            src=src,  # [E]
            dst=dst,  # [E]
            t=timestamps,  # [E]
            node_feats=None,  # not defined yet
            msg=edge_attr,  # [E, num_edge_features]
        )

        torch.save(self.collate([data]), self.processed_paths[0])


if __name__ == "__main__":
    original_df = load_csv_file("./data/original/cityD-dataset.csv")
    filtered_original_df = original_df[(original_df["d"] != 26) & (original_df["d"] < 60)]
    print(f"Number of unique users: {filtered_original_df["uid"].nunique()}")
    print(f"Number of unique locations: {filtered_original_df.drop_duplicates(subset=["x", "y"]).shape[0]}")

    df = load_csv_file("./data/raw/cityD-trajectory-dataset-preprocessed.csv")
    print(f"Number of unique users: {df["uid"].nunique()}")
    print(f"Number of unique locations: {df["cell_id"].nunique()}")

    data = UserLocationInteractionDataset(root="data", city_idx="D")
    data = data[0]
    print(f"Number of unique users in TemporalData object: {torch.unique(data.src).numel()}")
    print(f"Number of unique locations in TemporalData object: {torch.unique(data.dst).numel()}")
