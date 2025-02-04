import torch
import random
import numpy as np

from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *
import os


def set_seed(seed: int = 42):
    """시드(Seed)를 고정하여 실험 재현성을 보장합니다."""
    random.seed(seed)  # Python random 시드 고정
    np.random.seed(seed)  # Numpy 시드 고정
    torch.manual_seed(seed)  # PyTorch 시드 고정
    torch.cuda.manual_seed(seed)  # GPU 연산 시드 고정


if __name__ == "__main__":
    # 🔹 시드 설정 추가
    set_seed(1234)

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint_w2v = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint_w2v)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (추가) 기존 체크포인트가 있으면 로드
    if os.path.exists("checkpoint.pt"):
        print("Loading existing model checkpoint from 'checkpoint.pt'...")
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")

    g = torch.Generator()
    g.manual_seed(1234)  # DataLoader에서 재현성을 유지하기 위한 시드 설정

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, 
        generator=g
    )
    
    validation_loader = DataLoader(
        dataset["validation"], batch_size=batch_size, shuffle=True, 
        generator=g
    )

    # train
    for epoch in range(num_epochs):
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(
                data["verse_text"], 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)

            labels = data["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        preds = []
        label_list = []
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(
                    data["verse_text"], 
                    padding=True, 
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                logits = model(input_ids)
                label_list += data["label"].tolist()
                preds += logits.argmax(-1).cpu().tolist()

        macro = f1_score(label_list, preds, average='macro')
        micro = f1_score(label_list, preds, average='micro')
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {loss_sum/len(train_loader):.6f}, "
            f"F1(macro): {macro:.6f}, F1(micro): {micro:.6f}"
        )

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")
    print(f"Model checkpoint saved at 'checkpoint.pt'")
