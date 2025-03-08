import torch
from data_utils import build_vocab, load_data, text_pipeline, len_vocab
from model import SentimentRNN
import config


# 检查是否有 GPU
device = config.device

batch_size = config.batch_size
embedding_dim = config.embedding_dim
hidden_dim = config.hidden_dim

# 1. 加载模型
model = SentimentRNN(len_vocab(), embedding_dim, hidden_dim).to(device)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()  # 切换到推理模式
print("Model loaded and ready for inference.")

# 6. 预测单个文本情感
def predict_sentiment(model, text):
    model.eval()
    with torch.no_grad():
        # 确保与训练时相同的预处理
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long).unsqueeze(0).to(device)
        length = torch.tensor([processed_text.size(1)], dtype=torch.long).to(device)

        output = model(processed_text, length).squeeze().item()

        print("Model output (logit):", output)  # 输出模型的原始logit值
        return "Positive" if output > 0.5 else "Negative"

prompt1 = "This movie is fantastic and very enjoyable!"
prompt2 = "I hated the film. It was so boring and awful."

print(f"Prompt1 Sentiment: {predict_sentiment(model, prompt1)}")
print(f"Prompt2 Sentiment: {predict_sentiment(model, prompt2)}")
