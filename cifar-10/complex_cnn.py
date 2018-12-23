#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch  
import torchvision  
import torchvision.transforms as transforms  
from torch.autograd import Variable
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim

device = torch.device("cuda:10" if torch.cuda.is_available() else "cpu")
device_ids = [10,11,12,13]
total_epoch = 0
batch = 128
WORKERS = 16
epoch_times = 200



class Net(nn.Module):                   
    def __init__(self):      
        super(Net, self).__init__()  
        self.conv1 = nn.Conv2d(3,  96, 3,stride=1,padding=1)  
        self.conv2 = nn.Conv2d(96, 96, 3,stride=1,padding=1) 
        self.conv3 = nn.Conv2d(96, 96, 3,stride=2,padding=1) 
        self.conv4 = nn.Conv2d(96, 192, 3,stride=1,padding=1) 
        self.conv5 = nn.Conv2d(192,192, 3,stride=1,padding=1) 
        self.conv6 = nn.Conv2d(192,192, 3,stride=2,padding=1) 
        self.conv7 = nn.Conv2d(192,192, 3,stride=1,padding=0) 
        self.conv8 = nn.Conv2d(192,192, 1,stride=1,padding=0)
        self.conv9 = nn.Conv2d(192, 10, 1,stride=1,padding=0) 
        #self.pool = nn.AvgPool2d(0, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
  
    def forward(self, x):                  
        """pool pool and relu relu"""
        x = F.dropout2d(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))  
        x = self.conv3(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = F.relu(self.conv4(x))  
        x = F.relu(self.conv5(x))  
        x = self.conv6(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = F.relu(x) 
        x = F.relu(self.conv7(x))  
        x = F.relu(self.conv8(x))  
        x = F.relu(self.conv9(x))  
        x = self.pool(x) 
 
        return x  


def test(testloader, batch, net):

    class_correct = list(0. for i in range(10))   
    class_total = list(0. for i in range(10))   
    for data in testloader:
        images, labels = data 
        '''GPU'''
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(Variable(images))  
        outputs = outputs.view(len(labels),10)
        _, predicted = torch.max(outputs.data, 1)  
        c = (predicted == labels).squeeze()  
        for i in range(len(labels)):  
            label = labels[i].item()
            class_correct[label] += c[i].item() 
            class_total[label] += 1  
  
    tt = 0
    ct = 0
    for i in range(10):
        ct = ct+class_correct[i]
        tt = tt+class_total[i]
    
    print('Accuracy of the network on the %d test images: %.1f %%' % (tt,  
    100 *ct / tt))
    return 100*ct/tt


def epoch_train(epoch_times, trainloader, net,total_epoch):
    begin = time.time()
    for epoch in range(epoch_times): 
        total_epoch = total_epoch + 1
        running_loss = 0.0 
        for i, data in enumerate(trainloader, 0): 
            inputs, labels = data   
            '''GPU'''
            inputs = inputs.to(device)
            labels = labels.to(device)
        # wrap them in Variable  
            inputs, labels = Variable(inputs), Variable(labels)   
        # \u8981\u628a\u68af\u5ea6\u91cd\u65b0\u5f52\u96f6\uff0c\u56e0\u4e3a\u53cd\u5411\u4f20\u64ad\u8fc7\u7a0b\u4e2d\u68af\u5ea6\u4f1a\u7d2f\u52a0\u4e0a\u4e00\u6b21\u5faa\u73af\u7684\u68af\u5ea6 
            optimizer.zero_grad()                 
        # forward + backward + optimize        
            outputs = net(inputs)
            outputs = outputs.view(len(labels),10)
            loss = criterion(outputs, labels)
            loss.backward() # \u5f53\u6267\u884c\u53cd\u5411\u4f20\u64ad\u4e4b\u540e\uff0c\u628a\u4f18\u5316\u5668\u7684\u53c2\u6570\u8fdb\u884c\u66f4\u65b0\uff0c\u4ee5\u4fbf\u8fdb\u884c\u4e0b\u4e00\u8f6e
            optimizer.step()                        
            running_loss += loss.data.item()
            
        print("[epoch: %d ]====[loss:%6f]" % (total_epoch, running_loss * batch/ 50000.0))

    end = time.time()
    print('training time:%.2f s'%(end - begin))
    
    return net, total_epoch



transform = transforms.Compose(  
    [transforms.ToTensor(),  
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  
                                        download=False, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,  
                                          shuffle=True, num_workers=WORKERS)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,  
                                       download=False, transform=transform)  
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=WORKERS)              

net = Net()
net = nn.DataParallel(net, device_ids = device_ids)
# ???
net.to(device)


classes = ('plane', 'car', 'bird', 'cat',  
'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

criterion = nn.CrossEntropyLoss() 
 
"""training!!! """

correct_rate = list(0. for i in range(epoch_times))

print('training!')
for j in range(epoch_times):
    # if j<20:
    #     optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.99, 0.999))
    # elif j < 100:
    #     optimizer = optim.Adam(net.parameters(), lr=0.005, betas=(0.8, 0.999))
    # else :
    #     optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999))
    if j<20:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif j < 100:
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    else :
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.999)
        
        
    net, total_epoch = epoch_train(1, trainloader, net, total_epoch)
    
    correct_rate[j] = test(testloader, batch, net)

print('finish training!')

torch.save(net.state_dict(), 'ComplexModel_'+batch+"_"+epoch_times+'.pth')
#net.load_state_dict(torch.load('model1.pth')) 


import matplotlib.pyplot as plt

file = open("./complex_data.txt","w")
file.write(str(correct_rate))
file.close()


plt.plot(correct_rate)
plt.draw()
plt.savefig("./fig/complex_cnn_"+str(batch)+"_"+str(epoch_times)+".png")
plt.show()