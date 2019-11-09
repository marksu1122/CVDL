import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import hw1_5_train


LeNet = hw1_5_train.LeNet()
LeNet.load_state_dict(torch.load('LENET_params.pkl'))

#測試數據
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1)
labels = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
images = []
for inputs, label in testloader:
    images.append(inputs)

def test(num):
    image = images[num]
    image = image / 2 + 0.5 
    npimg = image.numpy()
    
    ans = LeNet(image)
    m = nn.Softmax()
    arr = m(ans).detach().numpy()
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(npimg[0], (1, 2, 0)))
    plt.subplot(1,2,2)
    plt.bar(labels,arr[0])
    plt.show()

