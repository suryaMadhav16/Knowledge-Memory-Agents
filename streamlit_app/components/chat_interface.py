"""
Chat interface component for RAG applications.
"""
import streamlit as st
from typing import Callable, List, Dict, Any, Optional

def display_chat_history(messages: List[Dict[str, Any]]):
    """
    Display chat history in a clean format.
    
    Args:
        messages: List of message dictionaries, each with 'role' and 'content' keys
    """
    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

def display_sources(sources: List[Dict[str, Any]]):
    """
    Display sources used for the response.
    
    Args:
        sources: List of source dictionaries with metadata
    """
    if sources:
        with st.expander("Sources", expanded=False):
            for i, source in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {source.get('title', 'Document')}")
                st.markdown(f"*{source.get('source', 'Unknown source')}*")
                st.markdown(source.get('content', 'No content available'))
                st.markdown("---")

def chat_interface(
    process_query_func: Callable[[str], Dict[str, Any]],
    system_name: str = "RAG System",
    placeholder: str = "Ask a question...",
    initial_message: Optional[str] = None
):
    """
    Display a chat interface for interacting with a RAG system.
    
    Args:
        process_query_func: Function to process user queries, should return dict with 'answer' and optional 'sources'
        system_name: Name of the system to display in the interface
        placeholder: Placeholder text for the input field
        initial_message: Optional initial message from the assistant
    """
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
        # Add initial message if provided
        if initial_message:
            st.session_state.chat_messages.append({"role": "assistant", "content": initial_message})
    
    # Display chat history
    display_chat_history(st.session_state.chat_messages)
    
    # Chat input
    user_query = st.chat_input(placeholder)
    
    # Process user input
    if user_query:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        
        # Display the user message
        st.chat_message("user").write(user_query)
        
        # Get response from the RAG system
        with st.chat_message("assistant"):
            with st.spinner(f"{system_name} is processing your query..."):
                try:
                    # Process the query
                    response = process_query_func(user_query)
                    
                    # Display the response
                    st.write(response["answer"])
                    
                    # Display sources if available
                    if "sources" in response:
                        display_sources(response["sources"])
                    
                    # Add assistant message to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response["answer"]
                    })
                except Exception as e:
                    error_message = f"Error processing your query: {str(e)}"
                    st.error(error_message)
                    
                    # Add error message to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })

def rag_chat_interface(
    rag_system,
    system_name: str = "RAG System",
    initial_message: Optional[str] = None
):
    """
    Specialized chat interface for RAG systems.
    
    Args:
        rag_system: RAG system object with an invoke() method
        system_name: Name of the system to display
        initial_message: Optional initial message to display
    """
    def process_query(query: str) -> Dict[str, Any]:
        result = rag_system.invoke(query)
        
        # Format sources from context if available
        sources = []
        if "context" in result:
            for doc in result["context"]:
                sources.append({
                    "title": doc.metadata.get("title", "Document"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content
                })
        
        return {
            "answer": result["answer"],
            "sources": sources
        }
    
    return chat_interface(
        process_query,
        system_name=system_name,
        placeholder="Ask a question about the documents...",
        initial_message=initial_message
    )
