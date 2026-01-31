from pathlib import Path

import torch


class Linear_QNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model_sequence = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model_sequence(x)

    def save(self, model_path='model/model.pth'):
        file_path= Path(model_path)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)


        torch.save(self.state_dict(), file_path.absolute())
