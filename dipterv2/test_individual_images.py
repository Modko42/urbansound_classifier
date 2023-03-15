import os

import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np

class NETWORK2(nn.Module):
    def __init__(self):
        super(NETWORK2, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        ).cuda()
        # 15488

        self.flatten = nn.Flatten()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear_layers = nn.Linear(128, 10)
        # 1936
        # self.classifier = nn.Conv2d(15488, 9, kernel_size=1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        # x = self.classifier(x)
        return x



net = torch.load("Z:/Egyetem/önlab2_msc/saved_models/model_20230315_1958.pth")

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

haveCuda = torch.cuda.is_available()

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image



net.eval()

classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"];

def get_classes(indexes,classes):
    result = ""
    for i in indexes:
        result += classes[i]+" "
    return result

test_directory = "E:/urbandsounds8k/spectograms/v25_4_sound_features_50_10000noBGnoise/train/mixed_audio/"
filenames = os.listdir(test_directory)

for f in filenames:

    image = image_loader(transform,os.path.join(test_directory,f))
    image = image.cuda()
    output = net(image)


    _ , predictions = torch.topk(output,k=4,dim=1)
    predictions = predictions.cpu().detach().numpy()
    print(f.ljust(20,' ')," | ",get_classes(predictions.squeeze(0).tolist(),classes))

