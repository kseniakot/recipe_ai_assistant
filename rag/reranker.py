import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CrossEncoderReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def compute_score(
        self, pairs: list[list[str]], normalize: bool = True
    ) -> list[float]:
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        scores = self.model(**inputs).logits.view(-1).float()
        if normalize:
            scores = torch.sigmoid(scores)
        return scores.tolist()
