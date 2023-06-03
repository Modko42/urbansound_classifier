import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as fun
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import datetime

from IPython.display import HTML
from torchsummary import summary

start_time = time.time()
train_times = []
val_times = []


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


print(torch.cuda.is_available())
haveCuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


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
            nn.Dropout(p=0.6),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        return self.cnn_layers(x)


transform = transforms.Compose([
    transforms.ToTensor()

])


def print_gpu_stats():
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


label_translator = [
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 9],
    [1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [2],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [2, 7],
    [2, 8],
    [2, 9],
    [3],
    [3, 4],
    [3, 5],
    [3, 6],
    [3, 7],
    [3, 8],
    [3, 9],
    [4],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [4, 9],
    [5],
    [5, 6],
    [5, 7],
    [5, 8],
    [5, 9],
    [6],
    [6, 7],
    [6, 8],
    [6, 9],
    [7],
    [7, 8],
    [7, 9],
    [8],
    [8, 9],
    [9]
]


def multihot_encoder(labels, dtype=torch.float32):
    second_stage_labels = []
    # print("\nOriginal labels:")
    for l in labels:
        # print(l,end='')
        second_stage_labels.append(label_translator[l])
    # print("\nSecond stage labels:")
    labels = []
    for l2 in second_stage_labels:
        # print(l2,end='')
        labels.append(l2)

    # [14,5,34,3]
    # [[1,2],[5],[4,6],[3]]
    # [0,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0]
    label_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    multihot_vectors = []
    for label_list in labels:
        multihot_vectors.append([1 if x in label_list else 0 for x in label_set])

    # print("\nThird stage labels:")
    # for l3 in multihot_vectors[:5]:
    # print(l3,end='')

    return torch.Tensor(multihot_vectors).to(dtype)


print("Initialization took %.2f seconds." % (time.time() - start_time))


def train():
    running_loss = 0.0
    net.train()

    for data in trainLoader:

        inputs, labels = data
        new_labels = multihot_encoder(labels)
        if haveCuda:
            inputs, new_labels = inputs.cuda(), new_labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, new_labels.type(torch.float))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        running_loss += loss.item()

    tr_loss = running_loss / len(trainLoader)
    global best_train_loss
    if tr_loss < best_train_loss:
        best_train_loss = tr_loss

    return tr_loss


def val():

    running_loss = 0.0
    correct = 0.0
    total = 0.0
    net.eval()

    for data in testLoader:
        inputs, labels = data
        new_labels = multihot_encoder(labels)
        if haveCuda:
            inputs, new_labels = inputs.cuda(), new_labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, new_labels.type(torch.float))
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    val_loss = running_loss / len(testLoader)

    val_corr = correct / total * 100

    global best_val_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(net, "Z:/Egyetem/önlab2_msc/saved_models/model_" + str(
            datetime.datetime.now().strftime("%Y%m%d_%H%M")) + "_" + str(round(val_loss, 3)) + "_" + str(
            epoch) + "batchsize" + str(batchsize_) + ".pth")

    return val_loss,val_corr


torch.manual_seed(42)

net = NETWORK2()

summary(net, input_size=(3, 256, 256))

if haveCuda:
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if haveCuda:
    net = net.cuda()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10)

numEpoch = 32
trLosses = []
valLosses = []
best_train_loss = 10
best_val_loss = 10
batchsize_ = 64
trainSet = torchvision.datasets.ImageFolder(root="E:/temp_location/original_temp_location/train/", transform=transform)
testSet = torchvision.datasets.ImageFolder(root="E:/temp_location/original_temp_location/test/", transform=transform)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchsize_, shuffle=True,
                                          generator=torch.Generator(device='cuda'))
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchsize_, shuffle=False,
                                         generator=torch.Generator(device='cuda'))

print_gpu_stats()
for epoch in range(numEpoch):
    start = time.time()
    tr_loss = train()
    val_loss, correct_percentage = val()

    trLosses.append(tr_loss)
    valLosses.append(val_loss)

    print("Epoch " + str(epoch + 1) + ": \ntrain loss: " + str(tr_loss) + "\nval loss: " + str(val_loss))
    scheduler.step()
    end = time.time()
    print("Epoch " + str(epoch + 1) + " finished in " + str(end - start) + " seconds.")

print("Best validation loss: " + str(best_val_loss))
