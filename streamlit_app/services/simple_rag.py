"""
Simple RAG implementation.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap

class SimpleRAG:
    """
    Basic RAG implementation with a linear retrieve-then-generate pipeline.
    """
    def __init__(self, retriever, llm, prompt=None):
        """
        Initialize the Simple RAG system.
        
        Args:
            retriever: Document retriever
            llm: Language model
            prompt: Chat prompt template (if None, uses a default template)
        """
        self.retriever = retriever
        self.llm = llm
        
        # Create a default prompt template for RAG if not provided
        if prompt is None:
            self.prompt = ChatPromptTemplate.from_template(
                """Answer the following question based only on the provided context:  
                
                Context: {context}
                
                Question: {question}
                
                Answer:"""
            )
        else:
            self.prompt = prompt
        
        # Format the context
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        # Create the RAG chain
        self.rag_chain = (
            RunnableMap({
                "context": (lambda x: x["question"]) | self.retriever | format_docs,
                "question": lambda x: x["question"]
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG system.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        # For evaluation purposes, we'll return both the answer and the retrieved documents
        retrieved_docs = self.retriever.invoke(query)
        answer = self.rag_chain.invoke({"question": query})
        
        return {
            "answer": answer,
            "context": retrieved_docs
        }
