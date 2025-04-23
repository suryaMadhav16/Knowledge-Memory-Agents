"""
LLM utilities for RAG applications.
"""
import os
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def get_openai_llm(
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0, 
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Get an OpenAI LLM instance.
    
    Args:
        model_name: Name of the OpenAI model to use
        temperature: Temperature parameter for the model
        api_key: OpenAI API key (optional, will use env var OPENAI_API_KEY if not provided)
        **kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor
        
    Returns:
        ChatOpenAI instance
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        **kwargs
    )

def create_prompt_template(template: str) -> ChatPromptTemplate:
    """
    Create a chat prompt template.
    
    Args:
        template: The prompt template string
        
    Returns:
        ChatPromptTemplate instance
    """
    return ChatPromptTemplate.from_template(template)

def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Get a default RAG prompt template.
    
    Returns:
        ChatPromptTemplate instance
    """
    template = """Answer the following question based only on the provided context:  
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    return create_prompt_template(template)
