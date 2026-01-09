from src.models.codebert import CodeBERTModel


class EmbeddingGenerator:
    """
    Generates semantic embeddings for code.
    """

    def __init__(self):
        self.model = CodeBERTModel()

    def generate(self, code: str):
        return self.model.embed(code)
