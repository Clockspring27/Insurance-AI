import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_chain import InsuranceRAGChain
from config import Config

# Page configuration
st.set_page_config(
    page_title="Insurance Helper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000 !important
    }
    .source-box strong {
        color: #1f77b4 !important;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = True  #
    
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'vs_manager' not in st.session_state:
        try:
            st.session_state.vs_manager = VectorStoreManager()
        except Exception as e:
            st.error(f"Failed to initialize Vector Store: {str(e)}")
            st.stop()
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = InsuranceRAGChain(st.session_state.vs_manager)
    
    if 'collection_created' not in st.session_state:
        st.session_state.collection_created = True 


def detect_query_intent(question: str) -> str:
    """
    Automatically detect the intent of the user's question
    
    Args:
        question: User's question
        
    Returns:
        Query mode string
    """
    question_lower = question.lower()
    
    # Check for add-on related queries
    addon_keywords = ['addon', 'add-on', 'rider', 'optional cover', 'additional cover', 
                      'recommend', 'should i take', 'which cover', 'extra protection']
    if any(keyword in question_lower for keyword in addon_keywords):
        return "addons"
    
    # Check for exclusion related queries
    exclusion_keywords = ['exclusion', 'not covered', 'does not cover', "doesn't cover", 
                         'what is excluded', 'not include', 'gap', 'missing']
    if any(keyword in question_lower for keyword in exclusion_keywords):
        return "exclusions"
    
    # Check for coverage related queries
    coverage_keywords = ['what is covered', 'coverage', 'what does it cover', 'included',
                        'protection', 'insured for', 'claim for']
    if any(keyword in question_lower for keyword in coverage_keywords):
        return "coverage"
    
    # Check for term explanation queries
    term_keywords = ['explain', 'what is', 'what does', 'meaning of', 'define', 
                    'idv', 'ncb', 'depreciation', 'premium', 'term']
    if any(keyword in question_lower for keyword in term_keywords):
        return "terms"
    
    # Default to general query
    return "general"


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def process_pdfs(file_paths: List[str]) -> bool:
    """Process PDFs and add to vector store"""
    try:
        with st.spinner("Processing PDFs..."):
            # Process all PDFs
            all_chunks, all_metadata = st.session_state.pdf_processor.process_multiple_pdfs(file_paths)
            
            if not all_chunks:
                st.error("No content extracted from PDFs")
                return False
            
            # Create collection if not exists
            if not st.session_state.collection_created:
                st.session_state.vs_manager.create_collection(recreate=False)
                st.session_state.collection_created = True
            
            # Add documents to vector store
            st.session_state.vs_manager.add_documents(all_chunks)
            
            st.success(f"Successfully processed {len(file_paths)} PDF(s) with {len(all_chunks)} chunks")
            return True
            
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return False


def display_message(message: Dict[str, Any]):
    """Display a chat message"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander(f"View {len(message['sources'])} Sources"):
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {source['index']}:</strong> {source['source_file']} (Page {source['page']})<br>
                        <strong>Section:</strong> {source['section_type']}<br>
                        <strong>Preview:</strong> {source['content_preview']}
                    </div>
                    """, unsafe_allow_html=True)


def sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        
        # Smart mode indicator
        st.markdown("#### Smart Mode")
        st.info("The system automatically detects your question type and uses the best search strategy!")
        
        st.markdown("---")
        
        # Show what queries trigger what modes
        with st.expander("How Smart Mode Works"):
            st.markdown("""
            **Exclusions Mode** 
            - "What's not covered?"
            - "What are the exclusions?"
            
            **Coverage Mode** 
            - "What is covered?"
            - "What does this policy include?"
            
            **Terms Explanation** 
            - "Explain IDV"
            - "What does NCB mean?"
            
            **General Mode** 
            - Everything else
            """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    sidebar()
    
    # Main header
    st.markdown('<div class="main-header">Insurance Helper</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding your insurance made easy</div>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask about your insurance..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})
        
        # Get response based on auto-detected query intent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Automatically detect query intent
                    detected_intent = detect_query_intent(prompt)

                    intent_name = {
                        "addons": "Add-ons Analysis",
                        "exclusions": "Exclusions Check",
                        "coverage": "Coverage Analysis",
                        "terms": "Term Explanation",
                        "general": "General Query"
                    }
                    
                    st.caption(f"{intent_name.get(detected_intent, 'General Query')}")
                    
                    # Route to appropriate query method
                    if detected_intent == "addons":
                        result = st.session_state.rag_chain.query_specific_section(
                            prompt, 
                            section_type="addons"
                        )
                    elif detected_intent == "exclusions":
                        result = st.session_state.rag_chain.query_specific_section(
                            prompt, 
                            section_type="exclusions"
                        )
                    elif detected_intent == "coverage":
                        result = st.session_state.rag_chain.query_specific_section(
                            prompt, 
                            section_type="coverage"
                        )
                    else:  # terms or general
                        result = st.session_state.rag_chain.query(prompt)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if "sources" in result and result["sources"]:
                        with st.expander(f"View {len(result['sources'])} Sources"):
                            for source in result["sources"]:
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {source['index']}:</strong> {source['source_file']} (Page {source['page']})<br>
                                    <strong>Section:</strong> {source['section_type']}<br>
                                    <strong>Preview:</strong> {source['content_preview']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", [])
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":

    try:
        Config.validate_config()
        main()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please ensure your .env file contains all required API keys.")
