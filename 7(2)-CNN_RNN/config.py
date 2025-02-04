from typing import Literal
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
d_model = 2048

# Word2Vec
window_size = 7
# method: Literal["cbow", "skipgram"] = "skipgram"
method: Literal["cbow", "skipgram"] = "cbow"
lr_word2vec = 1e-03
num_epochs_word2vec = 100

# GRU
hidden_size = d_model
num_classes = 4
lr = 1e-03
num_epochs = 100
batch_size = 2048