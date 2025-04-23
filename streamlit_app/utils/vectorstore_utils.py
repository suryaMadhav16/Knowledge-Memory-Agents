"""
Vector store utilities for RAG applications.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import chromadb
from chromadb.config import Settings

def split_documents(
    documents: List[Document], 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    separators: List[str] = ["\n\n", "\n", ".", " ", ""]
) -> List[Document]:
    """
    Split documents into chunks for processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        separators: List of separators to use for text splitting
        
    Returns:
        List of split document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
# In utils/vectorstore_utils.py
# In utils/vectorstore_utils.py - simplified create_vectorstore function
def create_vectorstore(
    documents: List[Document],
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """
    Create a Chroma vector store with the specified documents and embedding model.
    """
    # Create the directory if it doesn't exist
    import os
    import shutil
    
    # Completely remove the directory to avoid conflicts
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Removed existing directory: {persist_directory}")
    
    # Create a new directory
    os.makedirs(persist_directory, exist_ok=True)
    
    # Import here to ensure it's available
    from langchain_community.vectorstores import Chroma
    
    # Use from_documents with completely fresh directory
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Created vector store with {len(documents)} documents")
    return vectorstore



def load_vectorstore(
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Optional[Chroma]:
    """
    Load an existing Chroma vector store.
    
    Args:
        embedding_model: Embedding model to use
        persist_directory: Directory where the vector store is persisted
        collection_name: Name of the collection to load
        
    Returns:
        Chroma vector store if it exists, None otherwise
    """
    if not os.path.exists(persist_directory):
        print(f"Persist directory does not exist: {persist_directory}")
        return None
    
    # Initialize the client with persistence
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Check if collection exists
    collections = client.list_collections()
    if not any(c.name == collection_name for c in collections):
        print(f"Collection does not exist: {collection_name}")
        return None
    
    # Load the vector store
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    
    return vectorstore
# In utils/vectorstore_utils.py

def get_or_create_vectorstore(
    documents: List[Document],
    embedding_model: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents"
) -> Chroma:
    """
    Get existing vector store or create a new one if it doesn't exist.
    
    Args:
        documents: List of documents to add if creating a new store
        embedding_model: Embedding model to use
        persist_directory: Directory to persist the vector store
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store
    """
    from langchain_community.vectorstores import Chroma
    import os
    import shutil
    
    # Check if directory exists
    if os.path.exists(persist_directory):
        try:
            # Try to load the existing vector store
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            
            # Try a simple query to check if it works
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
            _ = retriever.get_relevant_documents("test")
            
            print(f"Successfully loaded existing vector store from {persist_directory}")
            return vectorstore
            
        except Exception as e:
            print(f"Could not load existing vector store ({str(e)}), recreating...")
            # If loading fails for any reason, delete the directory and recreate
            shutil.rmtree(persist_directory)
    
    # Create directory if it doesn't exist
        os.makedirs(persist_directory, mode=0o777, exist_ok=True)

    
    # Create a new vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Created new vector store in {persist_directory}")
    return vectorstore