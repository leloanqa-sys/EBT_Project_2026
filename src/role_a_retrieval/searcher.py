from typing import List
from src.common.schemas import CandidateFrame

class VectorSearcher:
    """
    Role A Vector Searcher interface.
    Loads FAISS index and performs vector similarity search on keyframes.
    """
    def __init__(self, index_path: str = None):
        self.index_path = index_path

    def search_by_vector(self, query_vector, top_k: int = 100) -> List[CandidateFrame]:
        """
        Executes fast ANN search using FAISS vector index.
        Returns top-K CandidateFrame objects with clip_score set.
        """
        # Placeholder for FAISS index search
        return []

    def search_by_text(self, text_query: str, top_k: int = 100) -> List[CandidateFrame]:
        """
        Encodes query text using CLIP text encoder and performs search.
        """
        return []
