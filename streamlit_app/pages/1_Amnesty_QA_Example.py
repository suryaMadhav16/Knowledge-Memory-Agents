import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.embedding_utils import get_embeddings
from utils.llm_utils import get_openai_llm
from utils.vectorstore_utils import split_documents, create_vectorstore, load_vectorstore
from utils.evaluation_utils import (
    prepare_evaluation_dataset,
    evaluate_rag_system,
    create_metric_comparison_chart,
    create_radar_chart,
    calculate_overall_scores
)

# Import RAG implementations
from services.simple_rag import SimpleRAG
from services.crag import CRAG
from services.adaptive_rag import AdaptiveRAG

# Import LangChain components
from langchain_core.documents import Document
from datasets import load_dataset
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset

# Import web search components
from langchain_community.tools.tavily_search import TavilySearchResults

# Page configuration
st.set_page_config(
    page_title="Amnesty QA Example",
    page_icon="📊",
    layout="wide"
)

# Constants
PERSIST_DIR = "./amnesty_qa_chroma_db"
COLLECTION_NAME = "amnesty_qa"
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/amnesty_qa_analysis.pkl")
DATASET_SIZE = 25  # Number of examples to load
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# Main title
st.title("RAG Evaluation with Amnesty QA Dataset")
st.markdown("""
This example demonstrates RAG evaluation using the Amnesty QA dataset from Amnesty International reports.
The example compares three RAG architectures: Simple RAG, CRAG, and Adaptive RAG.
""")

# OpenAI API Key handling - hide in expander
with st.expander("API Keys", expanded=False):
    api_key = st.text_input("Enter your OpenAI API Key", type="password", 
                            value=os.environ.get("OPENAI_API_KEY", ""))
    tavily_key = st.text_input("Enter your Tavily API Key (optional for web search)", type="password",
                               value=os.environ.get("TAVILY_API_KEY", ""))
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    
    if st.button("Save Keys"):
        st.success("API keys saved!")

# Function to initialize web search (if tavily key available)
def init_web_search():
    if os.environ.get("TAVILY_API_KEY"):
        try:
            return TavilySearchResults(k=3)
        except Exception as e:
            st.warning(f"Could not initialize web search: {e}")
    return None

# Function to load/create dataset
@st.cache_data(ttl=3600)
def load_amnesty_dataset(size=DATASET_SIZE):
    try:
        # Fixed: Add trust_remote_code=True parameter to load the dataset
        amnesty_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v1", trust_remote_code=True)
        
        # Process the dataset to extract questions, contexts, etc.
        questions = []
        contexts_list = []
        answers = []
        
        # Take only the specified number of samples
        sample_count = min(size, len(amnesty_dataset["eval"]))
        
        for i in range(sample_count):
            item = amnesty_dataset["eval"][i]
            questions.append(item["question"])
            contexts_list.append(item["contexts"][:3])  # Limit to 3 contexts per question
            answers.append(item["answer"])
        
        dataset = {
            "questions": questions,
            "contexts": contexts_list,
            "answers": answers
        }
        
        # Create Document objects for vector store
        documents = []
        for i, contexts in enumerate(contexts_list):
            for j, context in enumerate(contexts):
                doc = Document(
                    page_content=context,
                    metadata={
                        "source": f"amnesty_{i}_context_{j}",
                        "question_id": i
                    }
                )
                documents.append(doc)
        
        return dataset, documents
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

def get_vectorstore(documents):
    """Get or create vector store for the documents"""
    # Use consistent embedding model
    embedding_model = get_embeddings(
        embedding_type="huggingface", 
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # First time it will create, subsequent runs will load
    with st.spinner("Loading or creating vector store..."):
        from utils.vectorstore_utils import get_or_create_vectorstore
        vectorstore = get_or_create_vectorstore(
            documents,
            embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
    
    return vectorstore, embedding_model
def init_rag_models(retriever):
    # Get LLM
    llm = get_openai_llm(model_name=OPENAI_DEFAULT_MODEL)
    
    # Initialize web search
    web_search_tool = init_web_search()
    
    # Initialize RAG models
    simple_rag = SimpleRAG(retriever, llm)
    
    crag = CRAG(
        retriever,
        llm,
        web_search_tool=web_search_tool
    )
    
    adaptive_rag = AdaptiveRAG(
        retriever,
        llm,
        web_search_tool=web_search_tool
    )
    
    return {
        "llm": llm,
        "simple_rag": simple_rag,
        "crag": crag,
        "adaptive_rag": adaptive_rag,
    }

# Function to load previous analysis results
def load_analysis_results():
    try:
        if os.path.exists(ANALYSIS_FILE):
            with open(ANALYSIS_FILE, 'rb') as f:
                results = pickle.load(f)
                # Verify this is the correct analysis file for Amnesty QA
                if any(r.get('dataset') == 'Amnesty' for r in results):
                    st.success(f"Loaded Amnesty QA analysis from {ANALYSIS_FILE}")
                    return results
                elif not any('dataset' in r for r in results):
                    # For backward compatibility with old analysis files that don't have dataset marker
                    st.info("Loaded analysis file without dataset marker - assuming it's for Amnesty QA")
                    # Add the dataset marker
                    for r in results:
                        r['dataset'] = 'Amnesty'
                    return results
                else:
                    st.warning("Found analysis file, but it appears to be for a different dataset.")
                    return None
        else:
            st.info(f"No previous analysis file found at {ANALYSIS_FILE}")
        return None
    except Exception as e:
        st.warning(f"Could not load previous analysis: {e}")
        return None

# Function to save analysis results
def save_analysis_results(results):
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(ANALYSIS_FILE), exist_ok=True)
        
        # Add a marker to identify this as an Amnesty QA analysis
        for result in results:
            result['dataset'] = 'Amnesty'
        
        with open(ANALYSIS_FILE, 'wb') as f:
            pickle.dump(results, f)
        st.success(f"Analysis results saved to {ANALYSIS_FILE}!")
    except Exception as e:
        st.error(f"Could not save analysis results: {e}")
        import traceback
        st.error(traceback.format_exc())

# Start app initialization
if "app_initialized" not in st.session_state:
    with st.spinner("Initializing RAG evaluation..."):
        # Load dataset
        dataset, documents = load_amnesty_dataset()
        
        if dataset and documents:
            # Get vector store
            vectorstore, embedding_model = get_vectorstore(documents)
            
            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize models
            rag_models = init_rag_models(retriever)
            
            # Load any previous analysis
            analysis_results = load_analysis_results()
            
            # Store everything in session state
            st.session_state.dataset = dataset
            st.session_state.documents = documents
            st.session_state.retriever = retriever
            st.session_state.rag_models = rag_models
            st.session_state.analysis_results = analysis_results
            st.session_state.app_initialized = True
            
            st.success("Initialization complete! Dataset loaded and RAG models ready.")

# MAIN CONTENT AREA
if "app_initialized" in st.session_state:
    # Analysis section at the top
    st.header("RAG Performance Analysis")
    
    # Check if we have analysis results
    if st.session_state.analysis_results:
        # Display existing analysis
        all_results = st.session_state.analysis_results
        
        # Display header with dataset information
        st.write("### Amnesty QA Dataset Analysis")
        st.write("This analysis compares the performance of different RAG architectures on the Amnesty QA dataset.")
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Simple RAG")
            simple_result = next((r for r in all_results if r["name"] == "Simple RAG"), None)
            if simple_result:
                # Add marker for Amnesty dataset when saving
                if 'dataset' not in simple_result:
                    simple_result['dataset'] = 'Amnesty'
                if simple_result.get("dataset") == "Amnesty":
                    st.success("Amnesty QA Dataset Analysis")
                for metric, score in simple_result["results"].items():
                    st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        
        with col2:
            st.subheader("Corrective RAG (CRAG)")
            crag_result = next((r for r in all_results if r["name"] == "CRAG"), None)
            if crag_result:
                for metric, score in crag_result["results"].items():
                    st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        
        with col3:
            st.subheader("Adaptive RAG")
            adaptive_result = next((r for r in all_results if r["name"] == "Adaptive RAG"), None)
            if adaptive_result:
                for metric, score in adaptive_result["results"].items():
                    st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
        
        # Visualization tabs
        st.subheader("Performance Comparison")
        tabs = st.tabs(["Metrics Comparison", "Radar Chart", "Overall Scores"])
        
        with tabs[0]:
            comparison_chart = create_metric_comparison_chart(all_results)
            st.pyplot(comparison_chart)
        
        with tabs[1]:
            radar_chart = create_radar_chart(all_results)
            st.pyplot(radar_chart)
        
        with tabs[2]:
            overall_scores = calculate_overall_scores(all_results)
            st.dataframe(overall_scores)
            
            # Create a bar chart for overall scores
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x="System", 
                y="Overall Score", 
                data=overall_scores, 
                palette="viridis",
                ax=ax
            )
            plt.title("Overall RAG System Performance", fontsize=16)
            plt.xlabel("System", fontsize=12)
            plt.ylabel("Weighted Score", fontsize=12)
            plt.ylim(0, 1.0)
            
            # Add score values on top of bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 5),
                    textcoords="offset points"
                )
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Option to run new analysis
        if st.button("Run New Analysis", type="secondary"):
            st.session_state.run_new_analysis = True
    else:
        # No previous analysis, show option to run new analysis
        st.info("No previous analysis found. Run an evaluation to compare the RAG architectures.")
        if st.button("Run Evaluation", type="primary"):
            st.session_state.run_new_analysis = True
    
    # Run new analysis if requested
    if st.session_state.get("run_new_analysis", False):
        with st.spinner("Running evaluation on all RAG models (this may take a while)..."):
            try:
                # Create evaluation LLM
                evaluator_llm = LangchainLLMWrapper(
                    get_openai_llm(model_name=OPENAI_DEFAULT_MODEL)
                )
                
                # Number of questions to evaluate
                num_eval_questions = min(10, len(st.session_state.dataset["questions"]))
                
                # Process each model
                simple_eval_data = []
                crag_eval_data = []
                adaptive_eval_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(num_eval_questions):
                    progress = i / num_eval_questions
                    progress_bar.progress(progress)
                    status_text.text(f"Processing question {i+1}/{num_eval_questions}")
                    
                    question = st.session_state.dataset["questions"][i]
                    reference = st.session_state.dataset["answers"][i]
                    
                    # Process with SimpleRAG
                    simple_result = st.session_state.rag_models["simple_rag"].invoke(question)
                    simple_eval_data.append({
                        "user_input": question,
                        "response": simple_result["answer"],
                        "reference": reference,
                        "retrieved_contexts": [doc.page_content for doc in simple_result["context"]]
                    })
                    
                    # Process with CRAG
                    crag_result = st.session_state.rag_models["crag"].invoke(question)
                    crag_eval_data.append({
                        "user_input": question,
                        "response": crag_result["answer"],
                        "reference": reference,
                        "retrieved_contexts": [doc.page_content for doc in crag_result["context"]]
                    })
                    
                    # Process with AdaptiveRAG
                    adaptive_result = st.session_state.rag_models["adaptive_rag"].invoke(question)
                    adaptive_eval_data.append({
                        "user_input": question,
                        "response": adaptive_result["answer"],
                        "reference": reference,
                        "retrieved_contexts": [doc.page_content for doc in adaptive_result["context"]]
                    })
                
                progress_bar.progress(1.0)
                status_text.text("Creating evaluation datasets...")
                
                # Create evaluation datasets
                simple_eval_dataset = EvaluationDataset.from_list(simple_eval_data)
                crag_eval_dataset = EvaluationDataset.from_list(crag_eval_data)
                adaptive_eval_dataset = EvaluationDataset.from_list(adaptive_eval_data)
                
                status_text.text("Running evaluations...")
                
                # Run evaluations
                simple_results = evaluate_rag_system(simple_eval_dataset, evaluator_llm)
                simple_results["name"] = "Simple RAG"
                
                crag_results = evaluate_rag_system(crag_eval_dataset, evaluator_llm)
                crag_results["name"] = "CRAG"
                
                adaptive_results = evaluate_rag_system(adaptive_eval_dataset, evaluator_llm)
                adaptive_results["name"] = "Adaptive RAG"
                
                # Store results
                all_results = [
                    simple_results,
                    crag_results,
                    adaptive_results
                ]
                
                st.session_state.analysis_results = all_results
                st.session_state.run_new_analysis = False
                
                # Save results for future runs
                save_analysis_results(all_results)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("Evaluation completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
    
    # Testing section (at the bottom)
    st.markdown("---")
    st.header("Test RAG Models")
    
    # Sample questions
    sample_questions = st.session_state.dataset["questions"]
    selected_question = st.selectbox("Choose a question", sample_questions)
    
    custom_question = st.text_input("Or enter a custom question")
    if custom_question:
        selected_question = custom_question
    
    if selected_question and st.button("Test RAG Models"):
        with st.spinner("Processing query..."):
            try:
                # Process with SimpleRAG
                simple_result = st.session_state.rag_models["simple_rag"].invoke(selected_question)
                
                # Process with CRAG
                crag_result = st.session_state.rag_models["crag"].invoke(selected_question)
                
                # Process with AdaptiveRAG
                adaptive_result = st.session_state.rag_models["adaptive_rag"].invoke(selected_question)
                
                # Display results in tabs
                tabs = st.tabs(["Simple RAG", "CRAG", "Adaptive RAG"])
                
                with tabs[0]:
                    st.markdown("### Answer:")
                    st.write(simple_result["answer"])
                    with st.expander("Retrieved Sources"):
                        for i, doc in enumerate(simple_result["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"*{doc.metadata.get('source', 'Unknown')}*")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                
                with tabs[1]:
                    st.markdown("### Answer:")
                    st.write(crag_result["answer"])
                    with st.expander("Retrieved Sources"):
                        for i, doc in enumerate(crag_result["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"*{doc.metadata.get('source', 'Unknown')}*")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                
                with tabs[2]:
                    st.markdown("### Answer:")
                    st.write(adaptive_result["answer"])
                    with st.expander("Retrieved Sources"):
                        for i, doc in enumerate(adaptive_result["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"*{doc.metadata.get('source', 'Unknown')}*")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"Error testing RAG models: {str(e)}")
    
    # Footer with about the dataset
    st.markdown("---")
    with st.expander("About the Amnesty QA Dataset"):
        st.markdown("""
        The Amnesty QA dataset contains questions and answers based on reports from Amnesty International's research.
        It's a valuable resource for evaluating RAG systems on human rights related content.

        The data includes:
        - Questions about human rights situations
        - Context passages from Amnesty International reports
        - Ground truth answers

        For more information, visit [Amnesty QA on HuggingFace](https://huggingface.co/datasets/explodinggradients/amnesty_qa).
        """)
