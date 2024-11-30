import torch
from torchvision import models
from torch import nn

def load_model():
    device = torch.device('cpu')
    model = models.swin_v2_b(weights='DEFAULT')
    model.head = nn.Linear(in_features=model.head.in_features, out_features=2)  # Предполагается, что у вас два класса
    model.load_state_dict(torch.load('checkpoints/best.pth'))
    model.to(device)
    model.eval()
    return model