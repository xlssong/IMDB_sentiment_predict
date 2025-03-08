import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from data_utils import build_vocab, load_data, text_pipeline, len_vocab
from model import SentimentRNN
import config

device = config.device
# 超参数
batch_size = config.batch_size
embedding_dim = config.embedding_dim
hidden_dim = config.hidden_dim

num_epochs = 10

train_loader, test_loader = load_data(batch_size)
#########################################

# 2. 初始化模型
model = SentimentRNN(len_vocab(), embedding_dim, hidden_dim).to(device)

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        i = 0
        for labels, texts, length in train_loader:
            # i = i + 1
            labels = labels.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()
            outputs = model(texts, length).squeeze()
            # 在训练和预测中分别添加此代码片段
            #print("Sample text after text_pipeline:", texts)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # if i%50 == 0:
            #     print(f"outputs: {outputs}; labels:{labels}")

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {correct / total:.4f}")

train()

# 5. 评估模型
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for labels, texts, length in test_loader:
            labels = labels.to(device)
            texts = texts.to(device)
            outputs = model(texts, length).squeeze()
            #print("Sample text after text_pipeline:", texts)

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

evaluate()
#
# 保存模型
torch.save(model.state_dict(), "sentiment_model.pth")
