import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal
from config import *

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        self.linear = nn.Linear(window_size - 1, d_model, bias=False)
        

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        if self.method == "skipgram":
            self._train_skipgram(
                corpus, tokenizer, lr, num_epochs, criterion, optimizer
            )
        else: 
            self._train_cbow(
                corpus, tokenizer, lr, num_epochs, criterion, optimizer
            )

    def _train_cbow(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int,
        criterion,
        optimization
    ) -> None:
        # 구현하세요!
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        context, target = self.cbow_data(corpus["verse_text"])
        x = tokenizer(context, padding=False, return_tensors="pt", add_special_tokens=False)\
                    .input_ids.to(device)
        y = tokenizer(target, padding=False, return_tensors="pt", add_special_tokens=False)\
                    .input_ids.to(device)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = x[i:i+batch_size]
                y_ = y[i:i+batch_size]
                optimization.zero_grad()
                pred = self.forward(x_)
                loss = criterion(pred, y_)
                loss.backward()
                optimization.step()

                total_loss += loss.item()
                numbatches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int,
        criterion,
        optimization
    ) -> None:
        # 구현하세요!
        # 구현하세요!
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        context, target = self.skip_data(corpus["verse_text"])
        x = tokenizer(context, padding=False, return_tensors="pt", add_special_tokens=False)\
                    .input_ids.to(device)
        y = tokenizer(target, padding=False, return_tensors="pt", add_special_tokens=False)\
                    .input_ids.to(device)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = x[i:i+batch_size]
                y_ = y[i:i+batch_size]
                optimization.zero_grad()
                pred = self.backward(x_)
                loss = criterion(pred, y_)
                loss.backward()
                optimization.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 구현하세요!
    def forward(self, x):
        x = self.linear(x)
        x = self.weight(x)
        return x
    
    def backward(self, x):
        x = self.weight(x)
        x = self.linear(x)
        return x

    @staticmethod
    def pad(sentence, pad = "<PAD>", window_size = window_size):
        pad = [pad] * window_size
        return pad + sentence + pad
    
    def cbow_data(self, token_data):
        result = []

        for sentence in token_data:
            pad = self.pad(sentence)
            for i in range(self.window_size, len(pad) - self.window_size):
                context = (
                    pad[i - self.window_size : i] +
                    pad[i + 1: i + self.window_size + 1]
                )
                target = pad[i]
                result.append((context, target))
        
        return result


    def skip_data(self, token_data):
        result = []

        for sentence in token_data:
            pad = self.pad(sentence)
            for i in range(self.window_size, len(pad) - self.window_size):
                context = (
                    pad[i - self.window_size : i] +
                    pad[i + 1: i + self.window_size + 1]
                )
                target = pad[i]
                result.append((target, context))
        
        return result
