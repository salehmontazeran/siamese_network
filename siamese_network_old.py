import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from data_utils.dataset import ChangeDetectionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16

train_dir = "./data/train"
val_dir = "./data/val"
test_dir = "./data/test"

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_dataset = ChangeDetectionDataset(val_dir, transforms)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_dataset = ChangeDetectionDataset(val_dir, transforms)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)

test_dataset = ChangeDetectionDataset(val_dir, transforms)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

lr = 1e-6
num_epoches = 100


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(524288, 1024),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(1024),
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


net = Cnn()
if torch.cuda.is_available():
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        print(label.size(), output1.size(), output2.size())
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance.size())
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


loss_func = ContrastiveLoss()
l_his = []

for epoch in range(num_epoches):
    print("Epoch:", epoch + 1, "Training...")
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):

        image1s, image2s, labels = data["a"], data["b"], data["label"]

        image1s = image1s.to(device)
        image2s = image2s.to(device)
        labels = labels.to(device)

        image1s, image2s, labels = (
            Variable(image1s),
            Variable(image2s),
            Variable(labels.float()),
        )

        optimizer.zero_grad()
        f1 = net(image1s)
        f2 = net(image2s)
        loss = loss_func(f1, f2, labels)
        loss.backward()
        optimizer.step()
        if i % 5 == 4:
            l_his.append(loss.data[0])
        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
print("Finished Training")
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(l_his)
plt.xlabel("Steps")
plt.ylabel("Loss")
fig.savefig("plott2.png")
# torch.save(net.state_dict(), name)


# net.load_state_dict(torch.load(name))
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )
# test_set = custom_dset("./lfw", "./train.txt", transform, transform)
# test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)
# correct = 0
# total = 0
# for data in test_loader:
#     image1s, image2s, labels = data
#     if torch.cuda.is_available():
#         image1s = image1s.cuda()
#         image2s = image2s.cuda()
#         labels = labels.cuda()
#     image1s, image2s, labels = (
#         Variable(image1s),
#         Variable(image2s),
#         Variable(labels.float()),
#     )
#     f1 = net(image1s)
#     f2 = net(image2s)
#     dist = F.pairwise_distance(f1, f2)
#     dist = dist.cpu()
#     for j in range(dist.size()[0]):
#         if dist.data.numpy()[j] < 0.8:
#             if labels.cpu().data.numpy()[j] == 1:
#                 correct += 1
#                 total += 1
#             else:
#                 total += 1
#         else:
#             if labels.cpu().data.numpy()[j] == 0:
#                 correct += 1
#                 total += 1
#             else:
#                 total += 1
# print(
#     "Accuracy of the network on the train images: %d %%" % (100 * correct / total)
# )

# test_set = custom_dset("./lfw", "./test.txt", transform, transform)
# test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)
correct = 0
total = 0
for data in test_dataloader:
    image1s, image2s, labels = data["a"], data["b"], data["label"]

    image1s = image1s.to(device)
    image2s = image2s.to(device)
    labels = labels.to(device)

    image1s, image2s, labels = (
        Variable(image1s),
        Variable(image2s),
        Variable(labels.float()),
    )
    f1 = net(image1s)
    f2 = net(image2s)
    dist = F.pairwise_distance(f1, f2)
    dist = dist.cpu()
    for j in range(dist.size()[0]):
        if dist.data.numpy()[j] < 0.8:
            if labels.cpu().data.numpy()[j] == 1:
                correct += 1
                total += 1
            else:
                total += 1
        else:
            if labels.cpu().data.numpy()[j] == 0:
                correct += 1
                total += 1
            else:
                total += 1
print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
