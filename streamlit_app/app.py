import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="RAG Evaluation & Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("RAG Evaluation & Optimization Dashboard")
st.markdown("""
This application demonstrates RAG (Retrieval Augmented Generation) systems evaluation and optimization techniques.
It includes worked examples using different datasets and comparative analysis of various RAG architectures.
""")

# Check for API keys
openai_key = os.environ.get("OPENAI_API_KEY", "")
tavily_key = os.environ.get("TAVILY_API_KEY", "")

if not openai_key:
    st.warning("⚠️ No OpenAI API key found. Please add your API key in the example pages or set it as an environment variable.")

# Main dashboard content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Available Examples")
    st.write("""
    We have implemented two detailed examples of RAG evaluation with different datasets:
    
    ### 1. Amnesty QA Dataset
    A grounded QA dataset created from Amnesty International research reports. 
    Ideal for evaluating RAG systems on human rights related content.
    
    ### 2. HuggingFace Documentation
    A synthetic dataset created from HuggingFace documentation, 
    demonstrating RAG evaluation techniques with technical documentation.
    """)
    
    # Example selection buttons
    st.button("Go to Amnesty QA Example", type="primary", key="goto_amnesty", 
              on_click=lambda: st._rerun_with_location(location="Amnesty_QA_Example"))
    st.button("Go to HuggingFace Doc Example", type="primary", key="goto_hfdoc",
              on_click=lambda: st._rerun_with_location(location="HuggingFace_Doc_Example"))

with col2:
    st.subheader("🔍 RAG Architectures Compared")
    st.write("""
    In both examples, we evaluate three advanced RAG architectures:
    
    ### Simple RAG
    A basic implementation with a linear retrieval-generation pipeline.
    
    ### Corrective RAG (CRAG)
    An advanced architecture that adds a self-assessment loop to filter and grade retrieved documents 
    and optionally fallback to web search when necessary.
    st.image("https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/CRAG_workflow.png",
             caption="CRAG workflow diagram", use_column_width=True)
    ### Adaptive RAG
    A sophisticated system that dynamically routes queries through different processing paths 
    based on query classification, optimizing for different query types.
    """)
    
    # Add a visual comparison (placeholder)
    st.image("https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/RAG_workflow.png", 
             caption="RAG workflow diagram", use_column_width=True)

# Dashboard overview
st.subheader("📊 Key Evaluation Metrics")
st.markdown("""
We use the RAGAS framework to evaluate our RAG implementations across multiple dimensions:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Measures if the response contains only information from the retrieved documents |
| **Context Precision** | Evaluates if retrieved documents are relevant to the question |
| **Context Recall** | Assesses if all information needed is present in retrieved documents |
| **Answer Relevancy** | Measures how well the response addresses the user's question |
| **Answer Correctness** | Compares the response with reference answers for factual accuracy |
""")

# Getting started
st.subheader("🚀 Getting Started")
st.markdown("""
To get started:

1. Select one of the example pages from the links above
2. Each example will guide you through:
   - Loading and understanding the dataset
   - Viewing evaluation results across different RAG architectures
   - Testing the RAG models with your own questions
   - Analyzing performance metrics and visualizations

3. You'll need an OpenAI API key for the LLM functionality
4. A Tavily API key is optional for web search features in CRAG and Adaptive RAG
""")

# Footer
st.markdown("---")
st.markdown("""
### About this Application

This dashboard demonstrates research from the RAG Evaluation and Optimization notebook, 
which explores advanced RAG architectures and evaluation techniques.

**Resources:**
- [RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation)
- [LangGraph CRAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- [LangGraph Adaptive RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
""")
