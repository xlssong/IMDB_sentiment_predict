import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
embedding_dim = 128
hidden_dim = 256