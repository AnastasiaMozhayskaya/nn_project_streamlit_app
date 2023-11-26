
import torch
from torch import nn
from torchvision.models import resnet18

# Архитектура модели
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)

    def forward(self, x):
        return self.model(x)