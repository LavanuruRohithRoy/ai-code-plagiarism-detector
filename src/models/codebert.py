from transformers import AutoTokenizer, AutoModel
import torch


class CodeBERTModel:
    """
    Loads CodeBERT once and provides embeddings.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self.model = AutoModel.from_pretrained(
            "microsoft/codebert-base"
        )
        self.model.eval()  # inference mode

    def embed(self, code: str) -> torch.Tensor:
        """
        Convert code into a single embedding vector.
        """

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over tokens
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.squeeze(0)
