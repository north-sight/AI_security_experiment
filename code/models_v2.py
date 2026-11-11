import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. 源模型 A：课程官方预训练卷积网络 (lenet_mnist_model.pth)
# 结构：Conv(1->32, 3x3), Conv(32->64, 3x3), MaxPool(2), Dropout, FC(9216->128), FC(128->10)
# 参数量：1,199,882
# =========================================================================
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

# =========================================================================
# 2. 源模型 B：现代化紧凑卷积网络 NetB_Modern
# 结构：带批归一化 (BatchNorm) 的双层大卷积核网络，测试不同正则化下的迁移特征
# 参数量：76,114
# =========================================================================
class NetB_Modern(nn.Module):
    def __init__(self, with_norm=True):
        super(NetB_Modern, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# =========================================================================
# 3. 黑盒目标模型 1：四层深层卷积网络 TargetCNN_Deep
# 结构：Conv(1->32) -> Conv(32->48) -> MaxPool -> Conv(48->64) -> Conv(64->96) -> MaxPool -> FC(4704->128->10)
# 参数量：692,890
# =========================================================================
class TargetCNN_Deep(nn.Module):
    def __init__(self, with_norm=True):
        super(TargetCNN_Deep, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(96 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # 28 -> 14
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2) # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# =========================================================================
# 4. 黑盒目标模型 2：三隐层深度多层感知机 TargetMLP_Deep
# 结构：彻底去除空间卷积偏置，784 -> 384 -> 192 -> 64 -> 10，带 Dropout 抑制共线性
# 参数量：388,362
# =========================================================================
class TargetMLP_Deep(nn.Module):
    def __init__(self, with_norm=True):
        super(TargetMLP_Deep, self).__init__()
        self.with_norm = with_norm
        self.fc1 = nn.Linear(784, 384)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(192, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        return logits

# =========================================================================
# 5. 黑盒目标模型 3：小型残差网络 TargetResNet_Mini
# 结构：带跨层残差快捷连接 (Residual Shortcut Connections)
# 参数量：77,290
# 目的：检验对抗扰动在跨越残差直连网络时的传递阻尼效应
# =========================================================================
class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class TargetResNet_Mini(nn.Module):
    def __init__(self, with_norm=True):
        super(TargetResNet_Mini, self).__init__()
        self.with_norm = with_norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = ResBlock(32, 32, stride=1)
        self.layer2 = ResBlock(32, 64, stride=2) # 28 -> 14
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        if self.with_norm:
            x = (x - 0.1307) / 0.3081
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.linear(out)
        return logits
