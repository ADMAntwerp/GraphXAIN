import copy
import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, confusion_matrix
from utils.utils import set_seed, binary_accuracy


SEED = 42


class GCN(torch.nn.Module):
    """Graph Convolutional Network for Binary Classification"""

    def __init__(self, dim_in, dim_h=128):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, 1)
        self.best_model_state_dict = None
        self.best_valid_acc = 0.0
        self.best_epoch = 0

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return torch.sigmoid(h)

    def fit(self, data, epochs):
        set_seed(SEED)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index).squeeze()
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask].float())
            acc = binary_accuracy(out[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0:
                val_out = out[data.val_mask]
                val_loss = loss_fn(val_out, data.y[data.val_mask].float())
                val_acc = binary_accuracy(val_out, data.y[data.val_mask].float())
                if val_acc > self.best_valid_acc:
                    self.best_valid_acc = val_acc
                    self.best_epoch = epoch
                    self.best_model_state_dict = copy.deepcopy(self.state_dict())

                print(
                    f"Epoch {epoch:>4} | Train Loss: {loss:.3f} | Train Acc:"
                    f" {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | "
                    f"Val Acc: {val_acc*100:.2f}%"
                )

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index).squeeze()

        preds = out[data.test_mask]
        labels = data.y[data.test_mask].float()
        acc = binary_accuracy(preds, labels)

        preds_np = preds.cpu().numpy()
        pred_labels = (preds_np > 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        auc = roc_auc_score(labels_np, preds_np)
        confm = confusion_matrix(labels_np, pred_labels)

        return acc, auc, confm

    def restore_best_model(self):
        """Restores the best model that was saved during training."""
        if self.best_model_state_dict:
            self.load_state_dict(self.best_model_state_dict)
            print(
                f"Best GCN model restored from epoch {self.best_epoch} with validation accuracy: {self.best_valid_acc.item() * 100:.2f}%"
            )
        else:
            print("No best model found to restore.")
