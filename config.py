import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Insurance Helper RAG application"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_HOST")
    
    # Gemini Model Configuration
    GEMINI_MODEL = "gemini-2.5-flash" 
    GEMINI_TEMPERATURE = 0.2 
    GEMINI_MAX_OUTPUT_TOKENS = 2048
    
    # Embedding Configuration
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIMENSION = 768
    
    # PDF Processing Configuration
    CHUNK_SIZE = 1000  
    CHUNK_OVERLAP = 200  
    
    # Chunking separators optimized for insurance documents
    SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence breaks
        ", ",    # Clause breaks
        " ",     # Word breaks
        ""       # Character breaks
    ]
    
    # Qdrant Configuration
    COLLECTION_NAME = "insurance_documents"
    VECTOR_SIZE = EMBEDDING_DIMENSION
    DISTANCE_METRIC = "Cosine"
    
    # Retrieval Configuration
    TOP_K = 5  
    SIMILARITY_THRESHOLD = 0.7  
    
    # RAG Prompt Template
    RAG_PROMPT_TEMPLATE = """You are an expert insurance advisor helping users understand their insurance documents.
Use the following context from insurance documents to answer the user's question.
Provide clear, accurate information and explain insurance terms in simple language.

Context from insurance documents:
{context}

User Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, clearly state that
- Explain insurance jargon in plain language
- For add-on recommendations, compare coverage, costs, and benefits
- Highlight important exclusions or limitations
- Be precise about coverage details and conditions

Answer:"""

    # System Instructions for Gemini
    SYSTEM_INSTRUCTION = """You are an expert insurance advisor. Your role is to:
1. Help users understand complex insurance documents
2. Explain insurance terms and conditions in simple language
3. Provide recommendations for add-ons based on coverage gaps
4. Compare different coverage options clearly
5. Highlight important exclusions and limitations
6. Always prioritize accuracy and clarity

Remember: You provide informational guidance only, not professional insurance advice."""

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_keys = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("QDRANT_API_KEY", cls.QDRANT_API_KEY),
            ("QDRANT_URL", cls.QDRANT_URL)
        ]
        
        missing = [key for key, value in required_keys if not value]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please set them in your .env file"
            )
        
        return True
    
    @classmethod
    def get_qdrant_config(cls):
        """Returns Qdrant configuration dictionary"""
        return {
            "url": cls.QDRANT_URL,
            "api_key": cls.QDRANT_API_KEY,
            "collection_name": cls.COLLECTION_NAME,
            "vector_size": cls.VECTOR_SIZE,
            "distance": cls.DISTANCE_METRIC
        }
    
    @classmethod
    def get_chunking_config(cls):
        """Returns chunking configuration dictionary"""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "separators": cls.SEPARATORS
        }
    
    @classmethod
    def get_retrieval_config(cls):
        """Returns retrieval configuration dictionary"""
        return {
            "top_k": cls.TOP_K,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD
        }