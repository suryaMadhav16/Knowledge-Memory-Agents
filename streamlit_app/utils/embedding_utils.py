"""
Embedding utilities for RAG applications.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional, Union

def get_openai_embeddings(api_key: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Get OpenAI embeddings model.
    
    Args:
        api_key: OpenAI API key (optional, will use env var OPENAI_API_KEY if not provided)
        
    Returns:
        OpenAIEmbeddings instance
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    return OpenAIEmbeddings()

def get_huggingface_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Get HuggingFace embeddings model.
    
    Args:
        model_name: Name of the HuggingFace model to use for embeddings
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def get_embeddings(
    embedding_type: str = "openai", 
    api_key: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    Get embeddings model based on specified type.
    
    Args:
        embedding_type: Type of embeddings to use ('openai' or 'huggingface')
        api_key: API key for OpenAI (if using OpenAI embeddings)
        model_name: Name of the HuggingFace model (if using HuggingFace embeddings)
        
    Returns:
        Embeddings model instance
    """
    if embedding_type.lower() == "openai":
        # return get_openai_embeddings(api_key)
        return get_huggingface_embeddings(model_name)
    elif embedding_type.lower() == "huggingface":
        return get_huggingface_embeddings(model_name)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
