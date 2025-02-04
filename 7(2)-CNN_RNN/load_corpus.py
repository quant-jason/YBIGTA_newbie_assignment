from datasets import load_dataset

# 구현하세요!

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    
    for split in ('train', 'validation', 'test'):
        for example in dataset[split]:
            corpus.append(example["verse_text"])
            
    return corpus

# def load_corpus() -> list[str]:
#     corpus: list[str] = []

#     poem_dataset = load_dataset("google-research-datasets/poem_sentiment")
#     for split in ("train", "validation", "test"):
#         for example in poem_dataset[split]:
#             corpus.append(example["verse_text"])
    
#     imdb_dataset = load_dataset("imdb")
#     for split in ("train", "test"):
#         for example in imdb_dataset[split]:
#             corpus.append(example["text"])

#     rt_dataset = load_dataset("rotten_tomatoes")
#     for split in ("train", "validation", "test"):
#         for example in rt_dataset[split]:
#             corpus.append(example["text"])

#     return corpus
