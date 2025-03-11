import torch
import torch.nn as nn

# 迁移到 MLU
device = torch.device("mlu")

# 定义 GAN 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # 归一化到 [-1, 1]，匹配 MNIST
        return x.view(-1, 1, 28, 28)  # 生成 28x28 灰度图

# 迁移到 MLU
generator = Generator().to(device)

# 生成随机噪声
z = torch.randn(64, 100).to(device)

# 生成图像
generated_images = generator(z)

print(f"Generated images shape on MLU: {generated_images.shape}")
