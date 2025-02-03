from datasets import load_dataset

# 구현하세요!

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    for i in dataset:
        corpus.append(i["verse_text"])
    return corpus