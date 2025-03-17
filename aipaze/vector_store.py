import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union

class VectorStore:
    """
    Simple vector store for semantic search and retrieval.
    """
    def __init__(self, name: str, embedding_model: str = "default"):
        """
        Initialize vector store.
        
        Args:
            name: Name of the vector store
            embedding_model: Embedding model to use (or "default")
        """
        self.name = name
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.embedding_model = embedding_model
        self.embedding_function = self._get_embedding_function(embedding_model)
    
    def _get_embedding_function(self, model_name: str) -> Callable:
        """
        Get embedding function based on model name.
        
        Args:
            model_name: Name of embedding model
            
        Returns:
            Function that converts text to embeddings
        """
        # Try to load sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            if model_name == "default":
                model_name = "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            return lambda texts: model.encode(texts, convert_to_numpy=True)
        except ImportError:
            logging.warning("sentence-transformers not installed. Using dummy embeddings.")
            # Fallback to dummy embeddings if library not available
            return lambda texts: np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dictionaries
        """
        if not documents:
            return
            
        # Generate embeddings
        new_embeddings = self.embedding_function(documents)
        
        # Add documents and embeddings
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        
        # Add metadata if provided
        if metadata:
            if len(metadata) != len(documents):
                raise ValueError("Length of metadata must match length of documents")
            self.metadata.extend(metadata)
        else:
            # Add empty metadata if not provided
            self.metadata.extend([{} for _ in documents])
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document, score, and metadata
        """
        if not self.documents:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_function(query)
        
        # Handle different embedding shapes
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        # Convert embeddings to numpy array if not already
        embeddings_array = np.array(self.embeddings)
        
        # Calculate similarity scores
        scores = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def save(self, directory: str):
        """
        Save vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save documents and metadata
        with open(os.path.join(directory, f"{self.name}_documents.json"), 'w') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "embedding_model": self.embedding_model
            }, f)
        
        # Save embeddings
        np.save(os.path.join(directory, f"{self.name}_embeddings.npy"), np.array(self.embeddings))
        
        logging.info(f"Vector store saved to {directory}")
    
    @classmethod
    def load(cls, directory: str, name: str) -> 'VectorStore':
        """
        Load vector store from disk.
        
        Args:
            directory: Directory to load from
            name: Name of vector store
            
        Returns:
            Loaded VectorStore instance
        """
        # Load documents and metadata
        with open(os.path.join(directory, f"{name}_documents.json"), 'r') as f:
            data = json.load(f)
        
        # Load embeddings
        embeddings = np.load(os.path.join(directory, f"{name}_embeddings.npy"))
        
        # Create vector store
        vector_store = cls(name, data["embedding_model"])
        vector_store.documents = data["documents"]
        vector_store.metadata = data["metadata"]
        vector_store.embeddings = embeddings.tolist()
        
        return vector_store