import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的Q网络
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 输入：游戏的状态（例如 CartPole 中的杆子位置、速度等）
state = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 这是一个简单的示例，CartPole 的状态有4个数值

# 模型实例化
model = QNetwork(input_size=4, output_size=2)  # 4个输入（状态维度），2个输出（动作维度）

# 输出：每个动作的 Q 值
q_values = model(state)  # 输出形状是 (2,) 对应2个动作的 Q 值

# 选择 Q 值最大的动作
action = torch.argmax(q_values).item()

print(f'Selected action: {action}')
