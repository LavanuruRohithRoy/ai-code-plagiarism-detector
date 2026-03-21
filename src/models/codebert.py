from transformers import AutoTokenizer, AutoModel
import torch
import os


class CodeBERTModel:
    """
    Loads CodeBERT once and provides embeddings.
    Uses lazy singleton pattern to load only on first use.
    """
    _instance = None
    _tokenizer = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._tokenizer is None or self._model is None:
            print("Loading CodeBERT model from cache (or HuggingFace)...")
            local_files_only = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/codebert-base",
                    local_files_only=local_files_only,
                )
                self._model = AutoModel.from_pretrained(
                    "microsoft/codebert-base",
                    local_files_only=local_files_only,
                )
            except OSError:
                if not local_files_only:
                    raise

                print("Offline cache miss for CodeBERT, retrying with online download...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/codebert-base",
                    local_files_only=False,
                )
                self._model = AutoModel.from_pretrained(
                    "microsoft/codebert-base",
                    local_files_only=False,
                )
            self._model.eval()  # inference mode
            print("CodeBERT model loaded successfully!")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

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
