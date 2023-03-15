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
        )
        #15488


        self.flatten = nn.Flatten()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear_layers = nn.Linear(128, 9)
        #1936
        #self.classifier = nn.Conv2d(15488, 9, kernel_size=1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        #x = self.classifier(x)
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

data_transforms = transforms.Compose([
    transforms.ToTensor()
])


net.eval()
#image = image_loader(data_transforms,"C:/Users/beni1/Desktop/own_music/spec/2.png")
timelist = []
firstlist  = []



genres = 'BLUE CLASS COUNT DISCO HIPHP METAL POP REGAE ROCK'
genres = genres.split()
directory = "Z:/Egyetem/önlab2_msc/testing/new/mountainking/specs"
filenames = os.listdir(directory)

print(filenames)
sum = numpy.zeros(9)
output = numpy.zeros(9)
for f in filenames:
    if f[-3:] == 'png':
        image = image_loader(data_transforms, os.path.join(directory,f))
        image = image.cuda()
        output = net(image).cpu().detach().numpy()
        output = output - np.min(output)
        output = output / np.max(output)
        sum = sum+output
        max_index = np.argmax(output)
        firstlist.append(str(genres[max_index]).ljust(10))

sum = sum - np.min(sum)
sum = sum / np.max(sum)
h = 0
for x in np.nditer(sum):
    print(genres[h].ljust(12) + ": " + str(x))
    h = h + 1
