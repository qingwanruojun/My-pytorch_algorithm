import torch
import torch.nn as nn

# 迁移到 MLU
device = torch.device("mlu")

# 定义 LSTM 进行情感分析
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1])
        return output

# 迁移模型到 MLU
model = SentimentLSTM(vocab_size=1000, embed_size=64, hidden_size=128, output_size=2).to(device)

# 生成示例数据（2 句长度为 5 的句子）
texts = torch.randint(0, 1000, (100, 5)).to(device)

# 前向传播
outputs = model(texts)
predicted_class = torch.argmax(outputs, dim=1)

print(f"Predicted Sentiment on MLU: {predicted_class.cpu()}")
