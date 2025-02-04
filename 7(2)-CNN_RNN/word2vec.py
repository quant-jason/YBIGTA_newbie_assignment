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

        self.softmax = nn.LogSoftmax(dim = 1)

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
        x, y = self.cbow_data(corpus, tokenizer)
        self.to(device)
       
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = x[i:i+batch_size]
                y_ = y[i:i+batch_size]

                x_tensor = torch.tensor(x_, dtype=torch.long, device=device)
                y_tensor = torch.tensor(y_, dtype=torch.long, device=device)
                
                optimization.zero_grad()

                pred = self.cbow(x_tensor)  
                
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimization.step()

                total_loss += loss.item()
                num_batches += 1
            
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, y = self.skip_data(corpus, tokenizer)
        
        self.to(device)

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = x[i:i+batch_size]
                y_ = y[i:i+batch_size]

                x_tensor = torch.tensor(x_, dtype=torch.long, device=device)
                y_tensor = torch.tensor(y_, dtype=torch.long, device=device)
                
                optimization.zero_grad()

                pred = self.skipgram(x_tensor)  
                # print(pred.shape)
                # print(x_tensor.shape)
                # print(y_tensor.shape)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimization.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 구현하세요!
    def skipgram(self, x):
        x = self.embeddings(x)
        x = self.weight(x)
        # x = self.softmax(x)
        return x
    
    def cbow(self, x):
        x = self.embeddings(x).sum(dim=1)
        x = self.weight(x)
        # x = self.softmax(x)
        return x

    @staticmethod
    def pad(sentence, tokenizer,  window_size = window_size):
        pad = tokenizer.pad_token_id
        pad = [pad] * window_size
        return pad + tokenizer(sentence).input_ids[1:-1] + pad
    
    def cbow_data(self, token_data, tokenizer):
        x = []
        y = []

        for sentence in token_data:
            pad = self.pad(sentence, tokenizer)
            for i in range(self.window_size, len(pad) - self.window_size):
                context = (
                    pad[i - self.window_size : i] +
                    pad[i + 1: i + self.window_size + 1]
                )
                target = pad[i]
                x.append(context)
                y.append(target)
        
        return x, y


    def skip_data(self, corpus, tokenizer):
        x, y = [], []
        for sentence in corpus:
            pad_seq = self.pad(sentence, tokenizer, self.window_size)
            for i in range(self.window_size, len(pad_seq) - self.window_size):
                center = pad_seq[i]
                # 주변 단어 모두 추출
                left_context  = pad_seq[i - self.window_size : i]
                right_context = pad_seq[i + 1 : i + self.window_size + 1]

                for w in (left_context + right_context):
                    x.append(center)
                    y.append(w)
        return x, y
