from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import Document
from langchain_classic.callbacks.base import BaseCallbackHandler
from utils.vector_store import VectorStoreManager
from config import Config
class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming responses"""
    
    def __init__(self):
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM"""
        self.text += token
        print(token, end="", flush=True)


class InsuranceRAGChain:
    """RAG chain for insurance document Q&A"""
    
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        """
        Initialize RAG chain
        
        Args:
            vector_store_manager: Optional VectorStoreManager instance
        """
        # Initialize vector store manager
        self.vs_manager = vector_store_manager or VectorStoreManager()
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=Config.GEMINI_TEMPERATURE,
            max_output_tokens=Config.GEMINI_MAX_OUTPUT_TOKENS,
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template=Config.RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        print("RAG chain initialized")
    
    def create_qa_chain(self, chain_type: str = "stuff") -> RetrievalQA:
        """
        Create a RetrievalQA chain
        
        Args:
            chain_type: Type of chain ("stuff", "map_reduce", "refine")
                       "stuff" - puts all docs in context (best for most cases)
                       
        Returns:
            RetrievalQA chain
        """
        retriever = self.vs_manager.get_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        return qa_chain
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional source documents
        """
        try:
            # Create QA chain
            qa_chain = self.create_qa_chain()
            
            # Run query
            result = qa_chain.invoke({"query": question})
            
            response = {
                "answer": result["result"],
                "question": question
            }
            
            if return_sources and "source_documents" in result:
                response["sources"] = self._format_sources(result["source_documents"])
                response["source_documents"] = result["source_documents"]
            
            return response
            
        except Exception as e:
            print(f" Error during query: {str(e)}")
            raise
    
    def query_with_context(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query with conversation context
        
        Args:
            question: User's question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            Dictionary with answer and sources
        """
        # Build contextualized question if history exists
        if conversation_history and len(conversation_history) > 0:
            context = "\n".join([
                f"Previous Q: {item['question']}\nPrevious A: {item['answer']}"
                for item in conversation_history[-3:]  # Last 3 turns
            ])
            contextualized_question = f"Conversation context:\n{context}\n\nCurrent question: {question}"
        else:
            contextualized_question = question
        
        return self.query(contextualized_question, return_sources=True)
    
    def query_specific_section(
        self, 
        question: str, 
        section_type: str
    ) -> Dict[str, Any]:
        """
        Query a specific section type (exclusions, addons, coverage, etc.)
        
        Args:
            question: User's question
            section_type: Section to search in
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Get relevant documents from specific section
            docs = self.vs_manager.search_by_section_type(
                query=question,
                section_type=section_type,
                k=5
            )
            
            if not docs:
                return {
                    "answer": f"No relevant information found in {section_type} section.",
                    "question": question,
                    "sources": []
                }
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "question": question,
                "sources": self._format_sources(docs),
                "source_documents": docs
            }
            
        except Exception as e:
            print(f"Error querying specific section: {str(e)}")
            raise
    
    def compare_addons(self, addon_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple add-ons
        
        Args:
            addon_names: List of add-on names to compare
            
        Returns:
            Dictionary with comparison and sources
        """
        question = f"Compare the following add-ons and explain their key differences, coverage, and benefits: {', '.join(addon_names)}"
        
        return self.query_specific_section(question, section_type="addons")
    
    def find_coverage_gaps(self, current_coverage_description: str) -> Dict[str, Any]:
        """
        Identify potential coverage gaps
        
        Args:
            current_coverage_description: Description of current coverage
            
        Returns:
            Dictionary with gap analysis and recommendations
        """
        question = f"""Based on this current coverage: {current_coverage_description}
        
        Please identify:
        1. What scenarios or risks are NOT covered
        2. What add-ons or riders could fill these gaps
        3. Which gaps are most important to address"""
        
        return self.query(question, return_sources=True)
    
    def explain_terms(self, terms: List[str]) -> Dict[str, Any]:
        """
        Explain insurance terms in plain language
        
        Args:
            terms: List of insurance terms to explain
            
        Returns:
            Dictionary with explanations
        """
        question = f"Explain these insurance terms in simple language: {', '.join(terms)}"
        
        return self.query(question, return_sources=True)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for display
        
        Args:
            documents: List of source documents
            
        Returns:
            List of formatted source information
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            source_info = {
                "index": i,
                "source_file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "section_type": doc.metadata.get("section_type", "general"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    
    def stream_query(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Query with streaming response
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, sources)
        """
        try:
            # Get relevant documents using invoke method
            retriever = self.vs_manager.get_retriever()
            docs = retriever.invoke(question)
            
            if not docs:
                return "No relevant information found in the documents.", []
            
            # Build context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Stream response
            print("\n Assistant: ", end="")
            stream_handler = StreamHandler()
            
            streaming_llm = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=Config.GEMINI_TEMPERATURE,
                streaming=True,
                callbacks=[stream_handler]
            )
            
            streaming_llm.invoke(prompt)
            print("\n")
            
            return stream_handler.text, self._format_sources(docs)
            
        except Exception as e:
            print(f" Error during streaming query: {str(e)}")
            raise


