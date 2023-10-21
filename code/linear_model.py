import torch.nn as nn

class LinearModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True) # outputs 4 values
        
    def forward(self, input):
        out = input.view(input.size(0), -1) # convert batch_size x 28 x 28 to batch_size x (28*28)
        out = self.fc(out) # Applies out = input * A + b. A, b are parameters of nn.Linear that we want to learn
        return out
