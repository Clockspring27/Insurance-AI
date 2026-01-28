from typing import List, Optional, Dict, Any
from langchain_classic.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import Config
import uuid


class VectorStoreManager:
    """Manages Qdrant vector store operations for insurance documents"""
    
    def __init__(self):
        """Initialize Qdrant client and embeddings"""
        # Validate configuration
        Config.validate_config()
        
        # Get configuration
        self.qdrant_config = Config.get_qdrant_config()
        self.retrieval_config = Config.get_retrieval_config()
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_config["url"],
            api_key=self.qdrant_config["api_key"],
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            output_dimensionality=Config.EMBEDDING_DIMENSION,
            google_api_key=Config.GEMINI_API_KEY
        )
        
        self.collection_name = self.qdrant_config["collection_name"]
        
        print("Vector store manager initialized")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            recreate: If True, delete existing collection and create new one
            
        Returns:
            Boolean indicating success
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if recreate:
                    print(f"⚠ Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    print(f" Collection '{self.collection_name}' already exists")
                    return True
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.qdrant_config["vector_size"],
                    distance=Distance.COSINE
                )
            )
            
            print(f" Created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f" Error creating collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add documents to Qdrant vector store
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of document IDs
        """
        try:
            print(f"Adding {len(documents)} documents to vector store...")
            
            # Ensure collection exists
            self.create_collection(recreate=False)
            
            # Initialize vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            
            # Add documents in batches
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Generate unique IDs for this batch
                batch_ids = [str(uuid.uuid4()) for _ in batch]
                
                # Add to vector store
                vector_store.add_documents(documents=batch, ids=batch_ids)
                all_ids.extend(batch_ids)
                
                print(f"   Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            print(f" Successfully added {len(documents)} documents")
            return all_ids
            
        except Exception as e:
            print(f" Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query string
            k: Number of results to return (default from config)
            filter_dict: Optional metadata filters (e.g., {"section_type": "exclusions"})
            
        Returns:
            List of most similar Documents
        """
        try:
            if k is None:
                k = self.retrieval_config["top_k"]
            
            # Initialize vector store for querying
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )

            if filter_dict:
                # Get more results than needed
                results = vector_store.similarity_search(query=query, k=k*3)
                
                # Filter by metadata
                filtered_results = []
                for doc in results:
                    match = True
                    for key, value in filter_dict.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(doc)
                    
                    # Stop when we have enough results
                    if len(filtered_results) >= k:
                        break
                
                return filtered_results[:k]
            else:
                results = vector_store.similarity_search(query=query, k=k)
                return results
            
        except Exception as e:
            print(f" Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores
        
        Args:
            query: Search query string
            k: Number of results to return
            score_threshold: Minimum similarity score (default from config)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            if k is None:
                k = self.retrieval_config["top_k"]
            
            if score_threshold is None:
                score_threshold = self.retrieval_config["similarity_threshold"]
            
            # Initialize vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            
            # Search with scores
            results = vector_store.similarity_search_with_score(query=query, k=k)
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            
            print(f" Found {len(filtered_results)} results above threshold {score_threshold}")
            return filtered_results
            
        except Exception as e:
            print(f" Error during similarity search with score: {str(e)}")
            raise
    
    def search_by_section_type(
        self, 
        query: str, 
        section_type: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search within a specific section type (e.g., 'exclusions', 'addons')
        
        Args:
            query: Search query string
            section_type: Type of section to search in
            k: Number of results to return
            
        Returns:
            List of Documents from specified section type
        """
        filter_dict = {"section_type": section_type}
        return self.similarity_search(query=query, k=k, filter_dict=filter_dict)
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
            
        except Exception as e:
            print(f" Error getting collection info: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection
        
        Returns:
            Boolean indicating success
        """
        try:
            self.client.delete_collection(self.collection_name)
            print(f" Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f" Error deleting collection: {str(e)}")
            return False
    
    def get_retriever(self, **kwargs):
        """
        Get a LangChain retriever object for use in chains
        
        Args:
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            VectorStoreRetriever object
        """
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        
        # Set default search kwargs
        search_kwargs = {
            "k": self.retrieval_config["top_k"]
        }
        search_kwargs.update(kwargs)
        
        return vector_store.as_retriever(search_kwargs=search_kwargs)
