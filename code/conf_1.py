import torch
import torch.nn as nn
import torchvision.models as models


class MLPModel_Configuration1(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(MLPModel_Configuration1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)