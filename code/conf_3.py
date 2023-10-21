import torch.nn as nn



class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)  # 4 outputs for the bounding box (x, y, width, height)
        )
    
    def forward(self, input):
        output = self.features(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

