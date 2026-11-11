import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 源模型 A：课程提供的预训练 CNN (lenet_mnist_model.pth)
# 结构：Conv(1->32, 3x3), Conv(32->64, 3x3), MaxPool(2), Dropout(0.25), FC(9216->128), Dropout(0.5), FC(128->10)
# 参数量：1,199,882
class NetA(nn.Module):
    def __init__(self, with_norm=True):
        super(NetA, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

# 2. 源模型 B：小型经典 LeNet 型网络
# 结构：Conv(1->6, 5x5), AvgPool(2), Conv(6->16, 5x5), AvgPool(2), FC(256->120), FC(120->84), FC(84->10)
# 参数量：44,426
class NetB(nn.Module):
    def __init__(self, with_norm=True):
        super(NetB, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# 3. 目标模型 1：三层卷积网络 TargetCNN
# 结构：Conv(1->24, 3x3, p=1), MaxPool(2), Conv(24->48, 3x3, p=1), MaxPool(2), Conv(48->64, 3x3, p=1), FC(3136->96), FC(96->10)
# 参数量：340,490
class TargetCNN(nn.Module):
    def __init__(self, with_norm=True):
        super(TargetCNN, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 24, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(48, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(3136, 96)
        self.fc2 = nn.Linear(96, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# 4. 目标模型 2：两隐层全连接网络 TargetMLP (无卷积归纳偏置)
# 结构：Flatten, FC(784->256), FC(256->128), FC(128->10)
# 参数量：235,146
class TargetMLP(nn.Module):
    def __init__(self, with_norm=True):
        super(TargetMLP, self).__init__()
        self.with_norm = with_norm
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
