import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from config import Config
import re

class PDFProcessor:
    """Handles PDF loading, parsing, and chunking for insurance documents"""
    
    def __init__(self):
        self.chunking_config = Config.get_chunking_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config["chunk_size"],
            chunk_overlap=self.chunking_config["chunk_overlap"],
            separators=self.chunking_config["separators"],
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF file and extract text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add source filename to metadata
            filename = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source_file"] = filename
                doc.metadata["total_pages"] = len(documents)
            
            print(f"Loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def extract_metadata(self, documents: List[Document]) -> Dict:
        """
        Extract useful metadata from insurance documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "total_pages": len(documents),
            "source_file": documents[0].metadata.get("source_file", "unknown"),
            "document_type": self._identify_document_type(documents),
        }
        
        return metadata
    
    def identify_document_type(self, documents: List[Document]) -> str:
        """
        Attempt to identify the type of insurance document
        
        Args:
            documents: List of Document objects
            
        Returns:
            String indicating document type
        """
        # Combine first few pages to identify document type
        sample_text = " ".join([doc.page_content for doc in documents[:3]]).lower()
        
        # Common insurance document keywords
        if "policy schedule" in sample_text or "policy document" in sample_text:
            return "policy_document"
        elif "proposal form" in sample_text:
            return "proposal_form"
        elif "claim" in sample_text:
            return "claim_form"
        elif "endorsement" in sample_text:
            return "endorsement"
        elif "add-on" in sample_text or "rider" in sample_text:
            return "addon_coverage"
        else:
            return "general_insurance"
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text from PDF
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        

        text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bPage\s+\d+/\d+\b', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks optimized for RAG retrieval
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects with enhanced metadata
        """
        # Clean text in all documents
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Enhance metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # Add context hints based on content
            content_lower = chunk.page_content.lower()
            
            # Identify important sections
            if any(keyword in content_lower for keyword in ["exclusion", "not covered", "does not cover"]):
                chunk.metadata["section_type"] = "exclusions"
            elif any(keyword in content_lower for keyword in ["coverage", "covered", "insured"]):
                chunk.metadata["section_type"] = "coverage"
            elif any(keyword in content_lower for keyword in ["premium", "cost", "price"]):
                chunk.metadata["section_type"] = "pricing"
            elif any(keyword in content_lower for keyword in ["add-on", "rider", "optional"]):
                chunk.metadata["section_type"] = "addons"
            elif any(keyword in content_lower for keyword in ["claim", "settlement"]):
                chunk.metadata["section_type"] = "claims"
            else:
                chunk.metadata["section_type"] = "general"
        
        print(f"Created {len(chunks)} chunks from {len(documents)} pages")
        return chunks
    
    def process_pdf(self, file_path: str) -> tuple[List[Document], Dict]:
        """
        Complete pipeline: Load, extract metadata, and chunk a PDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, metadata)
        """
        # Load PDF
        documents = self.load_pdf(file_path)
        
        # Extract metadata
        metadata = self.extract_metadata(documents)
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        return chunks, metadata
    
    def process_multiple_pdfs(self, file_paths: List[str]) -> tuple[List[Document], List[Dict]]:
        """
        Process multiple PDF files
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Tuple of (all_chunks, all_metadata)
        """
        all_chunks = []
        all_metadata = []
        
        for file_path in file_paths:
            try:
                chunks, metadata = self.process_pdf(file_path)
                all_chunks.extend(chunks)
                all_metadata.append(metadata)
            except Exception as e:
                print(f"✗ Failed to process {file_path}: {str(e)}")
                continue
        
        print(f"\n Processed {len(file_paths)} PDFs")
        print(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks, all_metadata
