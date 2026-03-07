from __future__ import annotations

import numpy as np

from src.storage.faiss_index import FaissIndex


class FaissSearch:
	"""Thin adapter around the storage FAISS index for pipeline usage."""

	def __init__(self, index: FaissIndex):
		self.index = index

	def top_similarity(self, vector: np.ndarray) -> float:
		distances, _ = self.index.search(vector, k=1)

		if len(distances) == 0:
			return 0.0

		return float(distances[0])
