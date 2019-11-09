import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#數據處理
def getData():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #取得訓練數據
    trainset = torchvision.datasets.CIFAR10(
        root='./data', #下載的數據存放的位置
        train=True, 
        download=True, 
        transform=transform #下載的數據要進行的格式轉換
    )    
    #對數據進行批處理
    trainloader = torch.utils.data.DataLoader( 
        trainset,     
        batch_size=64,  
        shuffle=True,  
        num_workers=2
    )

    #數據的類別
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    return trainloader, testloader, classes

#這個 model 就是一個 class 繼承 torch.nn.Module,要override __init__ 和 forward
class LeNet(nn.Module):   
    def __init__(self):
        # 繼承並使用LeNet的父類別(nn.Module)的初始化方法
        super(LeNet, self).__init__()   # == nn.Module.__init__()
        # nn.Conv2d return一個Conv2d class的一個對象，該類中包含forward函數的實現
        # 當調用self.conv1(input)的時候，就會調用該類的forward函數
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #定義該神經網絡的向前傳播函數，該函數必須定義，一旦定義成功，向後傳播函數也會自動生成（autograd）
    #定義 model 接收 input 時，data 要怎麼傳遞、經過哪些 activation function 等等
    def forward(self, x):
        #輸入x經過卷積conv1之後，經過激活函數ReLU，使用2x2的窗口進行Max pooling，然後更新到x。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #  return是一個Variable
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #view函數將張量x變形成一維的向量形式，總特徵數並不改變，為接下來的全連接作準備。
        #x = x.view(-1, self.num_flat_features(x)) 
        x = x.view(x.size()[0], -1)
        #輸入x經過全連接1，再經過ReLU激活函數，然後更新x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 計算準確率
def test_trainData(net, testloader): 
    correct, total = .0, .0
    for inputs, label in testloader:
        output = net(inputs)
        _, predicted = torch.max(output, 1) #  獲取分類結果
        total += label.size(0) # 記錄總個數
        correct += (predicted == label).sum() # 記錄分類正確的個數 
    return (float(correct) / total)*100

def train(MAX_EPOCH):
    net = LeNet()
    trainloader,testloader, classes = getData() #加載數據
    ceterion = nn.CrossEntropyLoss() #交叉熵損失
    optimizer = optim.Adam(net.parameters(), lr = 0.001, betas=(0.9, 0.99))
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_list = []
    trainAcc =[]

    for epoch in range(1,MAX_EPOCH+1):
        for step, (inputs, labels) in enumerate(trainloader):
            #得到data
            inputs = Variable(inputs)
            labels = Variable(labels)
            #將梯度初始化為零
            optimizer.zero_grad()
            #把 input 通過網絡往前傳（forward propagation），取得預測 output
            outputs = net(inputs)
            #pytorch裡邊反向傳播是要基於一個loss標量，loss.backward()，從中獲取grad，所以還必須得算一下loss標量出來才能進行反向傳播
            #計算 error（目標和預測結果的差距） ?????
            loss = ceterion(outputs, labels)
            # 把 error 往回傳(backward propagation),一一計算每個參數對此error的貢獻(取導數)==>即反向傳播求梯度
            loss.backward()
            #更新所有参数(梯度),(對 error 貢獻越多處罰越多)
            optimizer.step()
            if(MAX_EPOCH == 1 or step == len(trainloader)-1):
                loss_list.append(loss)
            
        #test()
        trainAcc.append(test_trainData(net, trainloader))

       
    #繪製loss變化曲線
    if(MAX_EPOCH != 1):
        fig = plt.figure() 
        acc = fig.add_subplot(2, 1, 1)
        acc.plot(trainAcc)
        acc.set_title("Accuracy")
        acc.set_xlabel("epoch")
        acc.set_ylabel("%")
        loss = fig.add_subplot(2, 1, 2)
        loss.plot(loss_list)
        loss.set_xlabel("epoch")
        loss.set_ylabel("loss")
    else:
        plt.plot(loss_list)
        plt.xlabel("iteration")
        plt.ylabel("loss")
    plt.show()
    return net


