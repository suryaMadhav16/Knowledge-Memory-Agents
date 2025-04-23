"""
Adaptive RAG implementation.
"""
from typing import Dict, Any, List, Optional, Literal, Union, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Try to import LangGraph components
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for environments without LangGraph
    pass

class QueryClassification(BaseModel):
    """Classification of query type for routing."""
    query_type: Literal["simple_factual", "complex_relational", "current_event", "beyond_knowledge"] = \
        Field(description="The type of the query")
    reasoning: str = Field(description="Explanation for the classification")

class AdaptiveRAGState(TypedDict):
    """State for Adaptive RAG workflow."""
    question: str
    query_type: str
    context: List[Document]
    answer: str

class AdaptiveRAG:
    """
    Adaptive RAG implementation with routing based on query type.
    Dynamically routes queries through different processing paths for optimal handling.
    """
    def __init__(self, retriever, llm, web_search_tool=None, use_langgraph=True):
        """
        Initialize the Adaptive RAG system.
        
        Args:
            retriever: Document retriever
            llm: Language model
            web_search_tool: Optional web search tool for current events
            use_langgraph: Whether to use LangGraph for implementation (if available)
        """
        self.retriever = retriever
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.use_langgraph = use_langgraph and 'StateGraph' in globals()
        
        # Create the LLM for structured outputs
        self.classifier_llm = llm.with_structured_output(QueryClassification)
        
        # Create the classifier prompt
        self.classifier_prompt = ChatPromptTemplate.from_template(
            """Classify the following question into one of these categories:
            
            - simple_factual: Straightforward factual questions that can be answered with basic information
            - complex_relational: Questions requiring understanding relationships between multiple entities
            - current_event: Questions about recent events, news, or time-sensitive information
            - beyond_knowledge: Questions beyond the scope of your knowledge base
            
            Question: {question}
            
            Provide your classification and explain your reasoning."""
        )
        
        # Create the generation prompt
        self.generate_prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context. 
            If the context doesn't contain enough information to provide a complete answer, say so.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Create chains
        self.classifier_chain = self.classifier_prompt | self.classifier_llm
        self.generate_chain = self.generate_prompt | self.llm | StrOutputParser()
        
        # Setup LangGraph if available and requested
        if self.use_langgraph:
            self.graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build the LangGraph for Adaptive RAG workflow.
        
        Returns:
            StateGraph: The compiled LangGraph
        """
        # Node functions
        def classify_query(state: AdaptiveRAGState) -> AdaptiveRAGState:
            """Classify the query to determine processing path."""
            question = state["question"]
            
            classification = self.classifier_chain.invoke({"question": question})
            
            return {"query_type": classification.query_type}
        
        def route_query(state: AdaptiveRAGState) -> Literal["simple_retrieval", "web_search", "direct_generation"]:
            """Route the query to the appropriate processing path."""
            query_type = state["query_type"]
            
            if query_type in ["simple_factual", "complex_relational"]:
                return "simple_retrieval"
            elif query_type == "current_event" and self.web_search_tool:
                return "web_search"
            else:  # beyond_knowledge or no web search available
                return "direct_generation"
        
        def simple_retrieval(state: AdaptiveRAGState) -> AdaptiveRAGState:
            """Retrieve documents from the vector store."""
            question = state["question"]
            context = self.retriever.invoke(question)
            return {"context": context}
        
        def web_search(state: AdaptiveRAGState) -> AdaptiveRAGState:
            """Retrieve information from web search."""
            if not self.web_search_tool:
                # Fall back to retrieval if web search is not available
                print("⚠️ Web search tool not available, falling back to retrieval")
                return simple_retrieval(state)
            
            question = state["question"]
            try:
                search_results = self.web_search_tool.invoke({"query": question})
                
                # Convert search results to documents
                web_docs = []
                for result in search_results:
                    web_docs.append(Document(
                        page_content=result["content"],
                        metadata={"source": result.get("url", "Unknown"), "title": result.get("title", "Unknown")}
                    ))
                
                # If web search didn't return anything, fall back to vector store
                if not web_docs:
                    return simple_retrieval(state)
                
                return {"context": web_docs}
            except Exception as e:
                print(f"⚠️ Web search error: {e}, falling back to retrieval")
                return simple_retrieval(state)
        
        def direct_generation(state: AdaptiveRAGState) -> AdaptiveRAGState:
            """Set up for generation without retrieval."""
            return {"context": []}
        
        def generate_answer(state: AdaptiveRAGState) -> AdaptiveRAGState:
            """Generate the final answer."""
            question = state["question"]
            context = state["context"]
            
            if context:
                context_text = "\n\n".join([doc.page_content for doc in context])
                answer = self.generate_chain.invoke({
                    "context": context_text, 
                    "question": question
                })
            else:
                # Direct generation without context
                answer = self.llm.invoke([
                    ("system", "You are a helpful assistant. If you don't know the answer, say so."),
                    ("human", question)
                ]).content
                
            return {"answer": answer}
        
        # Build the graph
        workflow = StateGraph(AdaptiveRAGState)
        
        # Add nodes
        workflow.add_node("classify", classify_query)
        workflow.add_node("simple_retrieval", simple_retrieval)
        workflow.add_node("web_search", web_search)
        workflow.add_node("direct_generation", direct_generation)
        workflow.add_node("generate", generate_answer)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify",
            route_query,
            {
                "simple_retrieval": "simple_retrieval",
                "web_search": "web_search",
                "direct_generation": "direct_generation"
            }
        )
        
        # Add remaining edges
        workflow.add_edge("simple_retrieval", "generate")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("direct_generation", "generate")
        workflow.add_edge("generate", END)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        return workflow.compile()
    
    def _invoke_with_langgraph(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the LangGraph workflow.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        # Initialize the state
        initial_state = {
            "question": query,
            "query_type": "",
            "context": [],
            "answer": ""
        }
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "context": final_state["context"]
        }
    
    def _invoke_without_langgraph(self, query: str) -> Dict[str, Any]:
        """
        Process a query using a step-by-step approach without LangGraph.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        # Step 1: Classify the query
        classification = self.classifier_chain.invoke({"question": query})
        query_type = classification.query_type
        
        # Step 2: Route based on query type
        context = []
        if query_type in ["simple_factual", "complex_relational"]:
            # Use vector store retrieval
            context = self.retriever.invoke(query)
        elif query_type == "current_event" and self.web_search_tool:
            # Use web search for current events
            try:
                search_results = self.web_search_tool.invoke({"query": query})
                
                # Convert search results to documents
                for result in search_results:
                    context.append(Document(
                        page_content=result["content"],
                        metadata={"source": result.get("url", "Unknown"), "title": result.get("title", "Unknown")}
                    ))
                
                # If web search didn't return anything, fall back to vector store
                if not context:
                    context = self.retriever.invoke(query)
            except Exception as e:
                print(f"Web search error: {e}, falling back to retrieval")
                context = self.retriever.invoke(query)
        
        # Step 3: Generate answer
        if context:
            context_text = "\n\n".join([doc.page_content for doc in context])
            answer = self.generate_chain.invoke({
                "context": context_text, 
                "question": query
            })
        else:
            # Direct generation without context for beyond_knowledge queries
            answer = self.llm.invoke([
                ("system", "You are a helpful assistant. If you don't know the answer, say so."),
                ("human", query)
            ]).content
        
        return {
            "answer": answer,
            "context": context
        }
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the Adaptive RAG system.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        if self.use_langgraph:
            return self._invoke_with_langgraph(query)
        else:
            return self._invoke_without_langgraph(query)
