"""ActrModule: MLP-based supervised learning agent for Snake game."""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActrModule(nn.Module):
    """MLP network for supervised learning on Snake game states.
    
    Uses an 11-element state representation and outputs 3 action logits.
    """
    
    def __init__(self, input_size=11, hidden_size=256, output_size=3, device=None):
        """Initialize the ActrModule.
        
        Args:
            input_size: Size of input state vector (default: 11)
            hidden_size: Size of hidden layer (default: 256)
            output_size: Number of action outputs (default: 3)
            device: Device to use (default: auto-detect cuda/cpu)
        """
        super(ActrModule, self).__init__()
        
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Build MLP network
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.to(self._device)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,)
            
        Returns:
            Output logits of shape (batch_size, output_size) or (output_size,)
        """
        return self.net(x)
    
    def predict(self, observation, deterministic=True):
        """Predict action from observation.
        
        Args:
            observation: State observation (numpy array or tensor)
            deterministic: If True, use argmax; if False, sample from softmax
            
        Returns:
            Discrete action (0, 1, or 2)
        """
        # Convert to tensor if needed
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self._device)
        else:
            obs_tensor = observation.to(self._device)
            
        # Add batch dimension if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            logits = self.forward(obs_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                # Sample from softmax distribution
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
                
        return action
    
    def save(self, path: str):
        """Save model weights and configuration.
        
        Args:
            path: Path to save the model
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": {
                "net": "mlp",
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size
            }
        }
        with open(path, "wb") as f:
            torch.save(payload, f)
        logger.info("Model saved to %s", path)
    
    def load(self, path: str, map_location=None):
        """Load model weights from file.
        
        Args:
            path: Path to load the model from
            map_location: Device mapping (default: None, uses model's device)
        """
        if map_location is None:
            map_location = self._device
            
        with open(path, "rb") as f:
            payload = torch.load(f, map_location=map_location)
            
        # Handle both dict payload and raw state_dict
        state = payload.get("state_dict", payload)
        self.load_state_dict(state)
        self.to(self._device)
        logger.info("Model loaded from %s", path)
    
    def train_supervised(self, states, actions, epochs=1, batch_size=32, lr=1e-3, device=None):
        """Train the model using supervised learning.
        
        Args:
            states: Array of state observations (N, input_size)
            actions: Array of actions (N,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            device: Device to use for training (default: model's device)
            
        Returns:
            Dict with training metadata
        """
        if device is None:
            device = self._device
            
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        
        # Create DataLoader
        dataset = TensorDataset(states_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.train()
        total_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                
                logits = self.forward(batch_states)
                loss = criterion(logits, batch_actions)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)
        
        self.eval()
        
        return {
            "epochs": epochs,
            "batches": total_steps
        }
