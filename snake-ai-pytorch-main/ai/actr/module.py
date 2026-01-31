"""Module implemented as a PyTorch nn.Module.

This class provides a small MLP policy compatible with the project's 11-element state.
It includes convenience helpers for prediction, supervised training (train_supervised), and
saving/loading model weights.
"""
from typing import Optional, Any, Dict, Iterable
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ActrModule(nn.Module):
    """A small MLP policy implemented as torch.nn.Module.

    Constructor arguments:
    - input_size: dimensionality of the observation (default 11)
    - hidden_sizes: sequence of hidden layer sizes (default (128, 128))
    - output_size: number of discrete actions (default 3)

    Public methods:
    - forward(x): returns raw logits
    - predict(observation, deterministic=True) -> int
    - save(path) -> None
    - load(path, map_location=None) -> None
    - train_supervised(states, actions, epochs=1, batch_size=32, lr=1e-3, device=None) -> dict
    """

    def __init__(self, input_size: int = 11, hidden_sizes: Iterable[int] = (128, 128), output_size: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for i, h in enumerate(tuple(hidden_sizes)):
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        # final layer returns logits (no softmax)
        self.net = nn.Sequential(*layers)

        # device will be chosen at call time if not set explicitly
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (unnormalized scores).

        Accepts float tensors of shape (batch, input_size) or (input_size,) for single sample.
        """
        return self.net(x)

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        """Predict a discrete action index from a single observation.

        observation: numpy array or torch tensor with shape (input_size,) or (1, input_size).
        Returns: int action index.
        """
        # Convert to tensor
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        # If single-dim, add batch dim
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        observation = observation.to(self._device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(observation)
            if deterministic:
                action = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            else:
                probs = torch.softmax(logits, dim=1)
                action = int(torch.multinomial(probs, num_samples=1).cpu().numpy()[0, 0])
        return action

    def save(self, path: str) -> None:
        """Save the model state_dict to `path`.

        The parent directory is created as needed. Uses torch.save with a file context manager.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"state_dict": self.state_dict(), "config": {"net": "mlp"}}
        try:
            with open(path, "wb") as f:
                torch.save(payload, f)
            logger.info("Saved model to %s", path)
        except Exception:
            logger.exception("Failed to save model to %s", path)
            raise

    def load(self, path: str, map_location: Optional[Any] = None) -> None:
        """Load model state from `path`.

        map_location: passed to torch.load to control device mapping.
        """
        if map_location is None:
            map_location = self._device
        try:
            with open(path, "rb") as f:
                payload = torch.load(f, map_location=map_location)
            state = payload.get("state_dict", payload)
            self.load_state_dict(state)
            self.to(self._device)
            logger.info("Loaded model from %s", path)
        except Exception:
            logger.exception("Failed to load model from %s", path)
            raise

    def train_supervised(self, states: Any, actions: Any, epochs: int = 1, batch_size: int = 32, lr: float = 1e-3, device: Optional[Any] = None) -> Dict[str, Any]:
        """Train the network on (states, actions) as a supervised classification task.

        This helper treats `actions` as class indices and optimizes CrossEntropyLoss.
        Inputs may be numpy arrays or torch tensors. Returns training metadata.
        """
        if device is None:
            device = self._device
        else:
            device = torch.device(device)
        self.to(device)

        # Convert inputs to tensors
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)

        dataset = TensorDataset(states, actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.train()
        total_steps = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                total_steps += 1
            logger.info("Epoch %d loss %f", epoch, epoch_loss)

        return {"epochs": epochs, "batches": total_steps}
