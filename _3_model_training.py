import argparse
from datetime import datetime

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

from _2b_dataset import UserLocationInteractionDataset


# -------- Auxiliary modules -------- #
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictorWithTime(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_time = torch.nn.Linear(1, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst, t):
        h = self.lin_src(z_src) + self.lin_dst(z_dst) + self.lin_time(t.to(z_src.dtype).unsqueeze(-1))
        h = h.relu()
        return self.lin_final(h)


# -------- TGN Model -------- #
class TGNModel(torch.nn.Module):
    def __init__(self, device, num_nodes, msg_dim, memory_dim=100, time_dim=100, embedding_dim=100, neighbor_size=10):
        super().__init__()
        self.device = device
        self.neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)

        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        ).to(device)

        self.link_pred = LinkPredictorWithTime(in_channels=embedding_dim).to(device)
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def concatenate_message(self, user_feats, edge_feats, location_feats):
        return torch.cat([user_feats, edge_feats, location_feats], dim=-1)

    def reset_state(self):
        self.memory.reset_state()
        self.neighbor_loader.reset_state()

    def compute_embeddings(self, batch, data):
        n_id, edge_index, e_id = self.neighbor_loader(batch.n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        z, last_update = self.memory(n_id)
        msg = self.concatenate_message(data.user_feats[e_id], data.edge_feats[e_id], data.location_feats[e_id])
        z = self.gnn(z, last_update, edge_index, data.t[e_id].to(self.device), msg.to(self.device))
        return z

    def predict_scores(self, z, src, dst, t, neg_dst=None, neg_ratio=1.0):
        pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]], t)
        if neg_dst is not None:
            neg_src = src.repeat_interleave(int(neg_ratio))
            neg_t = t.repeat_interleave(int(neg_ratio))
            neg_out = self.link_pred(z[self.assoc[neg_src]], z[self.assoc[neg_dst]], neg_t)
            return pos_out, neg_out
        return pos_out, None

    def predict_inference_scores_for_single_user(self, z, src, candidate_dst, t):
        t = t.repeat_interleave(candidate_dst.size(0))
        scores = self.link_pred(z[self.assoc[src]], z[self.assoc[candidate_dst]], t)
        return scores

    def update_states(self, src, dst, t, user_feats, edge_feats, location_feats):
        msg = self.concatenate_message(user_feats, edge_feats, location_feats)
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)

    def detach_memory(self):
        self.memory.detach()


# -------- Training Functions -------- #
def train_epoch(model, train_loader, data, optimizer, criterion, epoch, writer=None):
    model.memory.train()
    model.gnn.train()
    model.link_pred.train()

    model.memory.reset_state()  # Start with a fresh memory.
    model.neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    total_events = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(model.device)

        z = model.compute_embeddings(batch, data)
        pos_out, neg_out = model.predict_scores(z, batch.src, batch.dst, batch.t, batch.neg_dst,
                                                train_loader.neg_sampling_ratio)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model.update_states(batch.src, batch.dst, batch.t, batch.user_feats, batch.edge_feats, batch.location_feats)

        loss.backward()
        optimizer.step()
        model.detach_memory()

        total_loss += float(loss) * batch.num_events
        total_events += batch.num_events

    avg_loss = total_loss / total_events
    if writer:
        writer.add_scalar("Loss/Train", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, data, epoch, split, writer=None):
    model.memory.eval()
    model.gnn.eval()
    model.link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(model.device)
        z = model.compute_embeddings(batch, data)
        pos_out, neg_out = model.predict_scores(z, batch.src, batch.dst, batch.t, batch.neg_dst,
                                                loader.neg_sampling_ratio)

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        model.update_states(batch.src, batch.dst, batch.t, batch.user_feats, batch.edge_feats, batch.location_feats)

    mean_ap = float(torch.tensor(aps).mean())
    mean_auc = float(torch.tensor(aucs).mean())
    if writer:
        writer.add_scalar(f"Metrics/{split}_AP", mean_ap, epoch)
        writer.add_scalar(f"Metrics/{split}_AUC", mean_auc, epoch)
    return mean_ap, mean_auc


# -------- Main with argparse -------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Graph Network for user-location link prediction.")
    parser.add_argument("--city", type=str, default="D", help="City index for dataset")
    parser.add_argument("--interpol", type=bool, default=True, help="Interpolation Yes/No")
    parser.add_argument("--small", type=bool, default=True, help="Only use users that will be predicted later")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs to wait without improvement before stopping.")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training")
    parser.add_argument("--neg_sampling_ratio", type=float, default=20.0, help="Negative sampling ratio")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--memory_dim", type=int, default=100, help="Dimension of TGN memory")
    parser.add_argument("--time_dim", type=int, default=100, help="Dimension of time encoding")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension of final node embeddings")
    parser.add_argument("--neighbor_size", type=int, default=10, help="Number of recent neighbors to store")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model training on device: {device}", flush=True)

    dataset = UserLocationInteractionDataset(root="data", city_idx=args.city, interpol=args.interpol, small=args.small)
    data = dataset[0].to(device)

    train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size,
                                      neg_sampling_ratio=args.neg_sampling_ratio)
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=args.neg_sampling_ratio)
    test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size, neg_sampling_ratio=args.neg_sampling_ratio)

    model = TGNModel(
        device=device,
        num_nodes=data.num_nodes,
        msg_dim=(data.user_feats.size(-1) + data.edge_feats.size(-1) + data.location_feats.size(-1)),
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embedding_dim=args.embedding_dim,
        neighbor_size=args.neighbor_size,
    )

    optimizer = torch.optim.Adam(
        set(model.memory.parameters()) | set(model.gnn.parameters()) | set(model.link_pred.parameters()),
        lr=args.lr
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"./model_training_runs/TGN-training_{timestamp}")

    best_test_ap = 0
    best_test_auc = 0
    no_improve_epochs = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, data, optimizer, criterion, epoch, writer)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}", flush=True)

        val_ap, val_auc = evaluate(model, val_loader, data, epoch, "Val", writer)
        print(f"Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}", flush=True)
        test_ap, test_auc = evaluate(model, test_loader, data, epoch, "Test", writer)
        print(f"Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}", flush=True)

        if test_ap > best_test_ap or test_auc > best_test_auc:
            no_improve_epochs = 0
            torch.save({
                "memory_state": model.memory.state_dict(),
                "gnn_state": model.gnn.state_dict(),
                "pred_state": model.link_pred.state_dict(),
                "memory_buffer": model.memory.memory.clone(),
                "last_update": model.memory.last_update.clone(),
                "neighbor_dict": model.neighbor_loader.neighbors.clone(),
                "assoc": model.assoc.clone(),
                "epoch": epoch
            }, f"./model_training_runs/city{args.city}/best_model_{args.epochs}_on_{args.city}_{args.interpol}_{args.small}.pt")
            best_test_ap = test_ap
            best_test_auc = test_auc

        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"No improvement for {args.patience} epochs. Early stopping at epoch {epoch}.")
                break

    torch.save({
        "memory_state": model.memory.state_dict(),
        "gnn_state": model.gnn.state_dict(),
        "pred_state": model.link_pred.state_dict(),
        "memory_buffer": model.memory.memory.clone(),
        "last_update": model.memory.last_update.clone(),
        "neighbor_dict": model.neighbor_loader.neighbors.clone(),
        "assoc": model.assoc.clone(),
    }, f"./model_training_runs/city{args.city}/last_model_{args.epochs}_on_{args.city}_{args.interpol}_{args.small}.pt")

    writer.close()
