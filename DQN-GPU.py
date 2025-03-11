import torch
import torch.nn as nn

# 选择 MLU 设备
device = torch.device("mlu")

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型迁移到 MLU
model = QNetwork(input_size=4, output_size=2).to(device)

# 输入状态数据并迁移到 MLU（保持数据类型一致）
state = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=device).unsqueeze(0)  # 添加 batch 维度

# 计算 Q 值
q_values = model(state)

# 选择 Q 值最大的动作，并迁移回 CPU
action = torch.argmax(q_values, dim=1).cpu().item()

print(f'Selected action on MLU: {action}')
