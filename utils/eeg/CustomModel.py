import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_size),
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out