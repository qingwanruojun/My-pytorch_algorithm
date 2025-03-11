import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设有一个简单的LSTM模型
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])  # 使用最后一个隐藏状态进行分类
        return output

# 输入：文本数据，已经通过词汇表转换成了整数索引序列
# 假设一个批次的输入是 [1, 23, 45, 67]（文本中单词的索引）
texts = torch.tensor([[1, 23, 45, 67], [2, 34, 56, 78]])  # batch_size=2, 每条文本4个词

# 模型实例化
model = SentimentLSTM(vocab_size=1000, embed_size=64, hidden_size=128, output_size=2)

# 输出：情感分类结果（正面/负面）
outputs = model(texts)  # 输出形状为 (batch_size, output_size)，此处是 [2, 2]，表示正负情感的概率
predicted_class = torch.argmax(outputs, dim=1)  # 获取正面（1）或负面（0）预测标签

print(f'Predicted sentiment: {predicted_class}')
