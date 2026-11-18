"""
PathSnap - Multi-Layer Perceptron (MLP) Neural Network Model (Member 3)
======================================================================
Defines the PyTorch binary classification neural network for Model 5.

Estimates the probability P(arc_i is ground truth | GPS-arc candidate features)
to serve as emission probabilities for subsequent HMM route reconstruction.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class BinaryMLPClassifier(nn.Module):
    """
    Configurable PyTorch Multi-Layer Perceptron for binary candidate classification.

    Architecture:
    - Input layer (input_dim matching engineered candidate feature count)
    - 2 to 3 fully connected hidden layers with ReLU activations
    - Dropout regularization after each hidden layer
    - Output layer producing a single scalar logit (unbounded real number)
    - Compatible with torch.nn.BCEWithLogitsLoss for numerical stability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        if len(hidden_dims) < 2 or len(hidden_dims) > 4:
            raise ValueError(f"hidden_dims should contain 2 to 4 layer sizes, got {hidden_dims}")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Output projection layer (single logit)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Kaiming/He normal initialization to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits.
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes calibrated probability P(y=1 | x) via Sigmoid activation.
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            probs: Tensor of shape (batch_size, 1) with values in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_config(self) -> dict:
        """Returns model hyperparameter configuration."""
        return {
            "model_name": "BinaryMLPClassifier",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        }
