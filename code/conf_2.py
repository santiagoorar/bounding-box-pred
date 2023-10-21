import torch
import torch.nn as nn
import torchvision.models as models


# let´s try different numbers of hidden dimensions in the layers and edit leakyrelu

class MLPModel_Configuration2(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(MLPModel_Configuration2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*3),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)