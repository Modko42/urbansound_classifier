import torch
import torchvision
from torch import nn
from torchvision import transforms


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
        self.flatten = nn.Flatten()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear_layers = nn.Linear(128, 9)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x


net = torch.load("Z:/Egyetem/önlab2_msc/saved_models/model_20220509_1721.pth")
transform = transforms.Compose([
    transforms.ToTensor()

])

testSet = torchvision.datasets.ImageFolder(root="E:/temp_location/test/", transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=128, shuffle=False)

haveCuda = torch.cuda.is_available()

percentage = [0] * 9
percentage2 = [0] * 9
percentage3 = [0] * 9

conf = torch.zeros(9, 9)
for i, data in enumerate(testLoader, 0):
    # get the inputs
    inputs, labels = data

    # Convert to cuda conditionally
    if haveCuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    # forward
    outputs = net(inputs)
    # compute statistics
    _, predicted3rd = torch.topk(outputs, k=3, dim=1)

    for label, [pred1, pred2, pred3] in zip(labels, predicted3rd):
        conf[label, pred1] += 1
        if label == pred1:
            percentage[label] = percentage[label] + 1
        if label == pred1 or label == pred2:
            percentage2[label] = percentage2[label] + 1
        if label == pred1 or label == pred2 or label == pred3:
            percentage3[label] = percentage3[label] + 1

print(conf / 9)
print("1th: " + str([round(x / 11, 3) for x in percentage]))
print("2nd: " + str([round(x / 11, 3) for x in percentage2]))
print("3rd: " + str([round(x / 11, 3) for x in percentage3]))
