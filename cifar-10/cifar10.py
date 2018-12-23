# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import time

device = torch.device("cuda:12" if torch.cuda.is_available() else "cpu")
device_ids = [12,13,14,15]
print(device)

BATCH_SIZE = 64
EPOCH = 20
INPUT_SIZE = 224
DATA_DIR = "./data"
WORKERS = 8

import torch.optim as optim



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# change the network to suit cifar-10
net = torchvision.models.vgg11()
net.classifier[6] = nn.Linear(4096, 10, bias = True)

net = nn.DataParallel(net, device_ids = device_ids)
net = net.to(device)


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train_data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
val_data_transforms =   transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=False, transform=train_data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)
testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                       download=False, transform=val_data_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

val_correct_rate = list(0 for i in range(EPOCH))

print("start to train:\n")

for epoch in range(EPOCH):  # loop over the dataset multiple times

    begin = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        '''GPU'''
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        if(epoch > 30 and epoch < 100):
            optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
            #optimizer.zero_grad()
        elif(epoch > 100):
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.99)
        
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if( i == 100 and epoch == 0):
            print("%d s in first %d imgs." % (time.time() - begin , i*BATCH_SIZE))
        if(i*BATCH_SIZE % 10000 == 0):
            print("===%d / 50000===" % (i*BATCH_SIZE))


    print('[epoch: %d] loss: %.3f' %
              (epoch + 1, running_loss /50000 ))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            '''GPU'''
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_correct_rate[epoch] = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    end = time.time()

    print("spend %d s\n" % (end - begin))

print('Finished Training')


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        '''GPU'''
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


import matplotlib.pyplot as plt


torch.save(net.state_dict(), "./models/cifar10_"+str(BATCH_SIZE)+"_"+str(EPOCH)+".pth")

file = open("./cifar10"+str(BATCH_SIZE)+"_"+str(EPOCH)+".txt","w")
file.write(str(val_correct_rate))
file.close()


plt.plot(val_correct_rate)
plt.savefig("./fig/cifar10"+str(BATCH_SIZE)+"_"+str(EPOCH)+".png")
plt.show()