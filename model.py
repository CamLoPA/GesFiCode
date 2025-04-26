import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
class FeatureNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FeatureNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  
        #self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512,6)
        
    def forward(self, x):       
        out = self.features(x) 
        #out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        #out = self.fc(out)
        return out