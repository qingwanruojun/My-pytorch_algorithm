import torch
import torch.nn as nn
import torch.optim as optim

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.tanh(self.fc2(x))  # 输出范围 [ -1, 1 ]，与图像像素匹配
        return x

# 输入：随机噪声（通常为100维向量）
z = torch.randn(64, 100)  # 批次大小为64，噪声向量大小为100

# 模型实例化
generator = Generator()

# 输出：生成的图像（假设生成28x28的图像）
generated_image = generator(z)  # 输出形状为 (batch_size, 784)，此处784对应28x28的像素展开

print(f'Generated image shape: {generated_image.shape}')
