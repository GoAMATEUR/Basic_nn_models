import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained ViT model and modify the classifier head
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)