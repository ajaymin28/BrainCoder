import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResnetFeatureRegressor(nn.Module):
    def __init__(self, num_features, output_size):
        super(ResnetFeatureRegressor, self).__init__()
        
        # Load a pre-trained CNN as the feature extractor
        weights = ResNet50_Weights.DEFAULT
        self.cnn = resnet50(weights=weights)
        
        # Remove the classification head of the CNN
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Freeze the weights of the CNN
        for param in self.cnn.parameters():
            param.requires_grad = True
        
        # Define a regression head
        self.fc = nn.Linear(num_features, output_size)

        self.preprocessin_fn = weights.transforms()
    
    def forward(self, x):
        # Extract features using the CNN
        features = self.cnn(x)
        
        # Flatten the features
        features = torch.flatten(features, 1)
        
        # Regression
        output = self.fc(features)
        return output