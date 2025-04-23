"""
Corrective RAG (CRAG) implementation.
"""
from typing import Dict, Any, List, Optional, Literal, Union, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for environments without LangGraph
    pass

class DocumentRelevanceCheck(BaseModel):
    """Assessment of document relevance to the question."""
    is_relevant: bool = Field(description="Whether the document is relevant to the question")
    reasoning: str = Field(description="Explanation of the relevance assessment")

class SearchDecision(BaseModel):
    """Decision on whether to use web search."""
    use_web_search: bool = Field(description="Whether to use web search")
    reasoning: str = Field(description="Explanation for the decision")

class CRAGState(TypedDict):
    """State for the CRAG workflow."""
    question: str
    retrieved_documents: List[Document]
    relevant_documents: List[Document]
    web_search_documents: List[Document]
    final_documents: List[Document]
    answer: str

class CRAG:
    """
    Corrective RAG (CRAG) implementation with document relevance checking and web search fallback.
    Based on the paper by Liu et al. and implemented with LangGraph when available.
    """
    def __init__(self, retriever, llm, web_search_tool=None, use_langgraph=True):
        """
        Initialize the CRAG system.
        
        Args:
            retriever: Document retriever
            llm: Language model
            web_search_tool: Optional web search tool for fallback
            use_langgraph: Whether to use LangGraph for implementation (if available)
        """
        self.retriever = retriever
        self.llm = llm
        self.web_search_tool = web_search_tool
        self.use_langgraph = use_langgraph and 'StateGraph' in globals()
        
        # Create the LLM for structured outputs
        self.structured_llm = llm.with_structured_output(DocumentRelevanceCheck)
        self.decision_llm = llm.with_structured_output(SearchDecision)
        
        # Create the document grader prompt
        self.grade_prompt = ChatPromptTemplate.from_template(
            """Assess whether the following document is relevant to answering the question.
            
            Question: {question}
            
            Document: {document}
            
            Provide a boolean indicating if the document is relevant and explain your reasoning."""
        )
        
        # Create the search decision prompt
        self.search_decision_prompt = ChatPromptTemplate.from_template(
            """Based on the retrieved documents and the question, decide if a web search is needed.
            
            Question: {question}
            
            Retrieved documents: {documents}
            
            Number of relevant documents: {num_relevant}
            
            Should we use web search to get better information? Provide a boolean decision and explain your reasoning."""
        )
        
        # Create the generation prompt
        self.generate_prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context. 
            If the context doesn't contain enough information to provide a complete answer, say so.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Create the query rewriting prompt for web search
        self.query_rewrite_prompt = ChatPromptTemplate.from_template(
            """Rewrite the following question to make it more effective for a web search. 
            Remove any personal context, clarify any ambiguous terms, and focus on the core information need.
            
            Original Question: {question}
            
            Rewritten Search Query:"""
        )
        
        # Create the chains for document grading, web search, and generation
        self.grader_chain = self.grade_prompt | self.structured_llm
        self.decision_chain = self.search_decision_prompt | self.decision_llm
        self.query_rewrite_chain = self.query_rewrite_prompt | self.llm | StrOutputParser()
        self.generate_chain = self.generate_prompt | self.llm | StrOutputParser()
        
        # Setup LangGraph if available and requested
        if self.use_langgraph:
            self.graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build the LangGraph for CRAG workflow.
        
        Returns:
            StateGraph: The compiled LangGraph
        """
        # Define node functions
        def retrieve(state: CRAGState) -> CRAGState:
            """Retrieve documents from the vector store."""
            question = state["question"]
            retrieved_docs = self.retriever.invoke(question)
            return {"retrieved_documents": retrieved_docs}
        
        def grade_documents(state: CRAGState) -> CRAGState:
            """Grade each document for relevance."""
            question = state["question"]
            retrieved_docs = state["retrieved_documents"]
            
            relevant_docs = []
            for doc in retrieved_docs:
                try:
                    assessment = self.grader_chain.invoke({
                        "question": question, 
                        "document": doc.page_content
                    })
                    
                    if assessment.is_relevant:
                        relevant_docs.append(doc)
                except Exception as e:
                    print(f"Error grading document: {e}")
            
            return {"relevant_documents": relevant_docs}
        
        def decide_web_search(state: CRAGState) -> Literal["use_web_search", "skip_web_search"]:
            """Decide whether to perform web search."""
            if not self.web_search_tool:
                return "skip_web_search"
                
            question = state["question"]
            retrieved_docs = state["retrieved_documents"]
            relevant_docs = state["relevant_documents"]
            
            try:
                decision = self.decision_chain.invoke({
                    "question": question,
                    "documents": "\n\n".join([doc.page_content for doc in retrieved_docs]),
                    "num_relevant": len(relevant_docs)
                })
                
                if decision.use_web_search:
                    return "use_web_search"
                else:
                    return "skip_web_search"
            except Exception as e:
                print(f"Error in search decision: {e}")
                return "skip_web_search"
        
        def perform_web_search(state: CRAGState) -> CRAGState:
            """Perform web search to find additional documents."""
            question = state["question"]
            
            # Web search should be available here since we checked in decide_web_search
            try:
                # Rewrite the query for better web search results
                search_query = self.query_rewrite_chain.invoke({"question": question})
                
                # Perform web search
                search_results = self.web_search_tool.invoke({"query": search_query})
                
                # Convert search results to documents
                web_docs = []
                for result in search_results:
                    web_docs.append(Document(
                        page_content=result["content"],
                        metadata={"source": result.get("url", "Unknown"), "title": result.get("title", "Unknown")}
                    ))
                
                return {"web_search_documents": web_docs}
            except Exception as e:
                print(f"Error in web search: {e}")
                return {"web_search_documents": []}
        
        def combine_documents_with_web_search(state: CRAGState) -> CRAGState:
            """Combine relevant documents with web search results."""
            relevant_docs = state["relevant_documents"]
            web_docs = state["web_search_documents"]
            
            final_docs = relevant_docs + web_docs
            
            # If we have no relevant docs, fall back to the original retrieved docs
            if len(final_docs) == 0:
                final_docs = state["retrieved_documents"]
                
            return {"final_documents": final_docs}
        
        def combine_documents_without_web_search(state: CRAGState) -> CRAGState:
            """Prepare final documents without web search."""
            relevant_docs = state["relevant_documents"]
            
            # If we have no relevant docs, fall back to the original retrieved docs
            if len(relevant_docs) == 0:
                final_docs = state["retrieved_documents"]
            else:
                final_docs = relevant_docs
                
            return {"final_documents": final_docs}
        
        def generate_answer(state: CRAGState) -> CRAGState:
            """Generate the final answer based on the final documents."""
            question = state["question"]
            final_docs = state["final_documents"]
            
            context_text = "\n\n".join([doc.page_content for doc in final_docs])
            answer = self.generate_chain.invoke({
                "context": context_text, 
                "question": question
            })
            
            return {"answer": answer}
        
        # Build the graph
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("perform_web_search", perform_web_search)
        workflow.add_node("combine_documents_with_web_search", combine_documents_with_web_search)
        workflow.add_node("combine_documents_without_web_search", combine_documents_without_web_search)
        workflow.add_node("generate_answer", generate_answer)
        
        # Add edges
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge based on web search decision
        workflow.add_conditional_edges(
            "grade_documents",
            decide_web_search,
            {
                "use_web_search": "perform_web_search",
                "skip_web_search": "combine_documents_without_web_search"
            }
        )
        
        workflow.add_edge("perform_web_search", "combine_documents_with_web_search")
        workflow.add_edge("combine_documents_with_web_search", "generate_answer")
        workflow.add_edge("combine_documents_without_web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
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
            "retrieved_documents": [],
            "relevant_documents": [],
            "web_search_documents": [],
            "final_documents": [],
            "answer": ""
        }
        
        # Run the workflow
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["answer"],
            "context": final_state["final_documents"]
        }
    
    def _invoke_without_langgraph(self, query: str) -> Dict[str, Any]:
        """
        Process a query using a step-by-step approach without LangGraph.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        # Step 1: Retrieve initial documents
        retrieved_docs = self.retriever.invoke(query)
        
        # Step 2: Grade each document for relevance
        relevant_docs = []
        for doc in retrieved_docs:
            try:
                assessment = self.grader_chain.invoke({
                    "question": query, 
                    "document": doc.page_content
                })
                
                if assessment.is_relevant:
                    relevant_docs.append(doc)
            except Exception as e:
                print(f"Error grading document: {e}")
        
        # Step 3: Decide if web search is needed
        web_docs = []
        if self.web_search_tool and len(relevant_docs) < 2:  # If we have fewer than 2 relevant docs
            try:
                decision = self.decision_chain.invoke({
                    "question": query,
                    "documents": "\n\n".join([doc.page_content for doc in retrieved_docs]),
                    "num_relevant": len(relevant_docs)
                })
                
                if decision.use_web_search:
                    # Rewrite the query for better web search results
                    search_query = self.query_rewrite_chain.invoke({"question": query})
                    
                    # Perform web search
                    search_results = self.web_search_tool.invoke({"query": search_query})
                    
                    # Convert search results to documents
                    for result in search_results:
                        web_docs.append(Document(
                            page_content=result["content"],
                            metadata={"source": result.get("url", "Unknown"), "title": result.get("title", "Unknown")}
                        ))
            except Exception as e:
                print(f"Error in web search: {e}")
        
        # Step 4: Combine documents
        if len(relevant_docs) > 0:
            final_docs = relevant_docs + web_docs
        else:
            # If we have no relevant docs, fall back to the original retrieved docs
            final_docs = retrieved_docs
            
        # Step 5: Generate the answer
        context_text = "\n\n".join([doc.page_content for doc in final_docs])
        answer = self.generate_chain.invoke({
            "context": context_text, 
            "question": query
        })
        
        return {
            "answer": answer,
            "context": final_docs
        }
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the CRAG system.
        
        Args:
            query: The user question as a string
            
        Returns:
            dict: A dictionary containing the answer and retrieved context
        """
        if self.use_langgraph:
            return self._invoke_with_langgraph(query)
        else:
            return self._invoke_without_langgraph(query)
