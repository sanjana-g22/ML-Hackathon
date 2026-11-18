"""Model 5 (MLP) run through the shared harness: same trip-grouped folds and student_test evaluation as the other models.

Uses Member 3's BinaryMLPClassifier network; only the training loop is wrapped so it fits/predicts like sklearn.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.mlp_model import BinaryMLPClassifier
from src.metrics import run_experiment

CONFIG = dict(hidden_dims=[128, 64, 32], dropout=0.2, lr=1e-3, weight_decay=1e-4,
              batch_size=4096, epochs=6, max_train_rows=300_000, seed=42)


class MLPCandidateClassifier:
    def __init__(self, hidden_dims, dropout, lr, weight_decay, batch_size, epochs, max_train_rows, seed):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_train_rows = max_train_rows
        self.seed = seed

    def _tensor(self, X):
        return torch.tensor(self.prep_.transform(X), dtype=torch.float32)

    def _logits(self, X_t):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), 65536):
                out.append(self.net_(X_t[i:i + 65536]).squeeze(1))
        return torch.cat(out)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        # random row sample per fit (same idea as the KNN run) so CPU training stays fast
        if len(X) > self.max_train_rows:
            keep = np.random.RandomState(self.seed).choice(len(X), self.max_train_rows, replace=False)
            X, y = X.iloc[keep], np.asarray(y)[keep]
        if X_val is not None and len(X_val) > self.max_train_rows:
            keep = np.random.RandomState(self.seed).choice(len(X_val), self.max_train_rows, replace=False)
            X_val, y_val = X_val.iloc[keep], np.asarray(y_val)[keep]
        # MLPs can't take NaN and need features on similar scales; fitted on training rows only
        self.prep_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        X_t = self._tensor(X)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        pos_weight = (y_t == 0).sum() / (y_t == 1).sum().clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.net_ = BinaryMLPClassifier(X_t.shape[1], hidden_dims=self.hidden_dims, dropout_rate=self.dropout)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if X_val is not None:
            Xv_t = self._tensor(X_val)
            yv_t = torch.tensor(np.asarray(y_val), dtype=torch.float32)

        self.history_ = {"train_loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            self.net_.train()
            order = torch.randperm(len(X_t))
            total = 0.0
            for i in range(0, len(X_t), self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(X_t[idx]).squeeze(1), y_t[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.history_["train_loss"].append(total / len(X_t))
            if X_val is not None:
                self.history_["val_loss"].append(loss_fn(self._logits(Xv_t), yv_t).item())
        return self

    def predict_proba(self, X):
        p = torch.sigmoid(self._logits(self._tensor(X))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(y):
    return MLPCandidateClassifier(**CONFIG)


def fit_with_history(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, X_va, y_va)
    print(f"    epoch losses train={np.round(model.history_['train_loss'], 4).tolist()} "
          f"val={np.round(model.history_['val_loss'], 4).tolist()}")
    return model.history_


if __name__ == "__main__":
    config = {**CONFIG, "loss": "BCEWithLogits (pos_weight=n_neg/n_pos)", "optimizer": "AdamW",
              "preprocessing": "median impute + StandardScaler"}
    run_experiment("mlp", "Neural network (MLP)", config, make_model, fit_fn=fit_with_history,
                   artifact_name="mlp_model.joblib")
