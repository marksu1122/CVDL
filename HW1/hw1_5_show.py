import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np




def imshow(img,labels):
    label_names = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure("Image")
    for j in range(10):
        image = img[j]
        image = image / 2 + 0.5  # unnormalization
        npimg = image.numpy()
        plt.subplot(1,10,j+1)
        #是因為plt.imshow在顯示時需要的輸入是（imgsize,imgsieze,channels)，但是這裡是（channels,imgsize,imgsieze)，所以需要將位置轉換
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.xlabel(label_names[labels[j]])
        plt.axis('on')
    
    plt.show()
    
def show():
    # 用來展示圖像的函數
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=10,  
        shuffle=True,  
        num_workers=2  
    )
    # 隨機得到一些訓練圖像
    dataiter = iter(trainloader) #生成迭代器
    images, labels = dataiter.next() #每次運行next()就會調用trainloader，獲得一個之前定義的batch_size大小的批處理圖片集
    imshow(images,labels) 

