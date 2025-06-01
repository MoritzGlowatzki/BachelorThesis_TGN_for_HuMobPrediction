from datetime import datetime

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastNeighborLoader, MeanAggregator

from II_b_dataset import UserLocationInteractionDataset


# -------- Auxiliary modules -------- #
class GraphAttentionEmbedding(torch.nn.Module):
    """
    Applies a TransformerConv layer over dynamic graph edges that uses multi-head self-attention on graph neighborhoods.
    - in_channels: dimension of node features (here, memory output dimension).
    - out_channels: desired embedding dimension (final size).
    - msg_dim: dimension of raw message features associated with each edge.
    - time_enc: Time encoding module from TGN memory, encodes time differences.
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        # x: node feature matrix [num_nodes, in_channels]
        # last_update: [num_nodes], last timestamp each node was updated
        # edge_index: [2, num_edges]
        # t: [num_edges] timestamps of edges
        # msg: [num_edges, msg_dim] raw edge features
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    """
    Given source and destination node embeddings, predicts probability of a link.
    Uses two linear transformations (for source and destination), sums them, applies ReLU,
    and final linear layer to output a single logit.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


# -------- TGN Model -------- #
class TGNModel(torch.nn.Module):
    """
    Temporal Graph Network (TGN) wrapper combining memory, GNN encoder, and link prediction.
    References: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py and
                Temporal Graph Networks for Deep Learning on Dynamic Graphs (Rossi et al. 2020)

    Encapsulates all components needed for temporal link prediction:
        1. Memory module: maintains node-specific embeddings updated over time.
        2. GraphAttentionEmbedding: GNN encoder updating embeddings from memory + local subgraph.
        3. LinkPredictor: scoring function for candidate edges.
        4. LastNeighborLoader: maintains a rolling window of recent neighbors per node.

    Args:
        device (torch.device): Device on which to place modules and tensors.
        num_nodes (int): Total number of nodes (users + locations) in the dynamic graph.
        msg_dim (int): Dimension of raw message features on edges.
        memory_dim (int): Dimension of internal memory vectors per node.
        time_dim (int): Dimension of time encoding vectors.
        embedding_dim (int): Dimension of final node embeddings after GNN.
        neighbor_size (int): Number of recent neighbors to keep per node for GNN subgraph.
    """

    def __init__(self, device, num_nodes, msg_dim, memory_dim=100, time_dim=100, embedding_dim=100, neighbor_size=10):
        super().__init__()
        self.device = device
        # Keeps track of most recent neighbors for each node
        self.neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)

        # TGNMemory stores a memory vector for each node
        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        ).to(device)

        # GNN encoder using GraphAttentionEmbedding
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        ).to(device)

        # Link predictor for source-destination pairs
        self.link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

        # Association tensor for mapping global to local node indices
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def reset_state(self):
        # Resets memory and neighbor loader states at epoch start
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def compute_embeddings(self, batch, data):
        # Prepare subgraph from set of node IDs involved in the current batch
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        # Map each node ID in n_id to its local index
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        # Get memory vectors and last update times
        z, last_update = self.memory(n_id)
        # Update embeddings via GNN
        z = self.gnn(z, last_update,
                     edge_index,
                     data.t[e_id].to(self.device),
                     data.msg[e_id].to(self.device))
        return z

    def predict_scores(self, z, src, dst, neg_dst=None, neg_ratio=1.0):
        # Positive score
        pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]])
        # Negative score (if neg_dst provided)
        if neg_dst is not None:
            neg_src = src.repeat_interleave(int(neg_ratio))
            neg_out = self.link_pred(z[self.assoc[neg_src]], z[self.assoc[neg_dst]])
            return pos_out, neg_out
        return pos_out, None

    def update_states(self, src, dst, t, msg):
        # Update memory and neighbor histories
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)

    def detach_memory(self):
        # Detach memory to truncate backpropagation
        self.memory.detach()


# -------- Training and Evaluation Functions -------- #
def train_epoch(model, train_loader, data, optimizer, criterion, epoch, writer=None):
    """
    Run one epoch of TGN training.
    """
    model.train()
    # Reset between epochs ensures no stale memory or neighbor state
    model.reset_state()

    total_loss = 0
    total_events = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(model.device)

        # Step 1: Compute current embeddings for nodes in this batch
        z = model.compute_embeddings(batch, data)

        # Step 2: Score positive and negative edges
        pos_out, neg_out = model.predict_scores(
            z, batch.src, batch.dst, batch.neg_dst, train_loader.neg_sampling_ratio)

        # Step 3: Compute loss: positives → label 1, negatives → label 0
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Step 4: Update memory and neighbor history before backpropagation
        model.update_states(batch.src, batch.dst, batch.t, batch.msg)

        # Step 5: Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.detach_memory()  # Clear computation history so that past states aren’t tracked across batches

        # Accumulate total loss (weighted by number of events)
        total_loss += float(loss) * batch.num_events
        total_events += batch.num_events

    # Compute average loss per event
    avg_loss = total_loss / total_events

    if writer is not None:
        writer.add_scalar('Loss/Train', avg_loss, epoch)

    return avg_loss


@torch.no_grad()
def evaluate(model, loader, data, epoch, split, writer=None):
    """
    Evaluate TGN model on validation or test set.
    """
    model.eval()
    model.reset_state()
    torch.manual_seed(12345)

    aps, aucs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(model.device)

            # Compute embeddings
            z = model.compute_embeddings(batch, data)

            # Compute scores
            pos_out, neg_out = model.predict_scores(
                z, batch.src, batch.dst, batch.neg_dst, loader.neg_sampling_ratio)

            # Concatenate predictions and labels
            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat([
                torch.ones(pos_out.size(0)),
                torch.zeros(neg_out.size(0))
            ], dim=0)

            # Compute metrics
            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

            # Update states
            model.update_states(batch.src, batch.dst, batch.t, batch.msg)

    mean_ap = float(torch.tensor(aps).mean())
    mean_auc = float(torch.tensor(aucs).mean())

    if writer is not None:
        writer.add_scalar(f'Metrics/{split}_AP', mean_ap, epoch)
        writer.add_scalar(f'Metrics/{split}_AUC', mean_auc, epoch)

    return mean_ap, mean_auc


# -------- Main Training Function -------- #
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model training on device: {device}")

    # Load temporal interaction dataset
    dataset = UserLocationInteractionDataset(root="data", city_idx="D")
    data = dataset[0].to(device)

    neg_sampling_ratio = 1.0

    # Split data and initialize data loaders
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    train_loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=neg_sampling_ratio)
    val_loader = TemporalDataLoader(val_data, batch_size=200, neg_sampling_ratio=neg_sampling_ratio)
    test_loader = TemporalDataLoader(test_data, batch_size=200, neg_sampling_ratio=neg_sampling_ratio)

    # Instantiate TGNModel
    model = TGNModel(
        num_nodes=data.num_nodes,
        msg_dim=data.msg.size(-1),
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        neighbor_size=10,
        device=device
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        set(model.memory.parameters()) | set(model.gnn.parameters()) | set(model.link_pred.parameters()),
        lr=0.0001
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f'./model_training_runs/TGN-training_{timestamp}')

    # Training loop
    for epoch in range(1, 51):
        loss = train_epoch(model, train_loader, data, optimizer, criterion, epoch, writer)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        val_ap, val_auc = evaluate(model, val_loader, data, epoch, 'Val', writer)
        test_ap, test_auc = evaluate(model, test_loader, data, epoch, 'Test', writer)
        print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

        # Save model checkpoint
        torch.save({
            'memory_state': model.memory.state_dict(),
            'gnn_state': model.gnn.state_dict(),
            'pred_state': model.link_pred.state_dict(),
            'epoch': epoch
        }, './model_training_runs/best_model.pt')

    writer.close()
