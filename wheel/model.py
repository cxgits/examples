import torch 
import torch.nn as nn

# 创建网络模型

# 卷积神经网络（两个卷积层）
class ConvNet(nn.Module): # nn,neural network
    def __init__(self, num_classes=10): #0~9种类别
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), #输出尺寸：(n+2p-f)/s +1 = (28+4-5)/1 + 1 = 28,输入28输出还是28 ,1*28*28
            nn.BatchNorm2d(16), # 输出通道16,16*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #最大池化，做下采样，f=2,s=2相当于图像减半 #图片变模糊，保留原图片的特征，让训练参数减少。 #16*14*14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), #输入16通道，输出32通道
            nn.BatchNorm2d(32), #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #32*7*7
        self.fc = nn.Linear(7*7*32, num_classes) #全连接层展开
        
    def forward(self, x): #前向传播
        out = self.layer1(x) #in bx1x28x28 out bx16x14x14
        out = self.layer2(out)#out bx32x7x7
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)#bx10
        return out