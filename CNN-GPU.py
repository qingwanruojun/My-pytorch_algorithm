import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 迁移到 MLU
device = torch.device("mlu")

# 预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型迁移到 MLU
model = CNN().to(device)

# 获取数据
images, labels = next(iter(trainloader))
images, labels = images.to(device), labels.to(device)

# 前向传播
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print(f"Predicted labels on MLU: {predicted.cpu()}")
