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
from langchain_core.prompts import PromptTemplate

# Import web search components
from langchain_community.tools.tavily_search import TavilySearchResults

# Page configuration
st.set_page_config(
    page_title="HuggingFace Doc Example",
    page_icon="📚",
    layout="wide"
)

# Constants
PERSIST_DIR = "./huggingface_doc_chroma_db"
COLLECTION_NAME = "huggingface_doc"
ANALYSIS_FILE = "./huggingface_doc_analysis.pkl"
DATASET_SIZE = 20  # Number of examples to load
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# Main title
st.title("RAG Evaluation with HuggingFace Documentation")
st.markdown("""
This example demonstrates RAG evaluation using a synthetic dataset created from HuggingFace documentation.
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
def load_hf_documentation_dataset(size=DATASET_SIZE):
    try:
        # Try to load the QA evaluation dataset first
        try:
            # Fixed: Add trust_remote_code=True parameter
            hf_dataset = load_dataset("m-ric/huggingface_doc_qa_eval", split="train", trust_remote_code=True)
            
            # Extract questions and contexts
            questions = []
            contexts = []
            answers = []
            
            # Limit to specified number of examples
            max_examples = min(size, len(hf_dataset))
            for i in range(max_examples):
                item = hf_dataset[i]
                questions.append(item["question"])
                # Some items might have multiple contexts
                if isinstance(item["source_doc"], list):
                    contexts.append(item["source_doc"])
                else:
                    contexts.append([item["source_doc"]])
                answers.append(item["answer"])
            
            # Get unique documents from contexts
            all_docs = []
            for context_list in contexts:
                all_docs.extend(context_list)
            unique_docs = list(set(all_docs))
            
            dataset = {
                "documents": unique_docs,
                "questions": questions,
                "contexts": contexts,
                "answers": answers
            }
            
            # Create Document objects for vector store
            documents = []
            for i, doc_text in enumerate(dataset["documents"]):
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "source": f"hf_doc_{i}",
                        "doc_id": i
                    }
                )
                documents.append(doc)
            
            return dataset, documents
        
        except Exception as e:
            # Fall back to the regular documentation dataset
            st.warning(f"Could not load QA evaluation dataset: {e}. Falling back to documentation dataset.")
            
            try:
                # Fixed: Add trust_remote_code=True parameter
                hf_dataset = load_dataset("m-ric/huggingface_doc", trust_remote_code=True)
                # Process the actual dataset
                documents_data = hf_dataset["train"]
                
                # Extract documents, limit to a reasonable number for demo
                docs_content = [
                    doc["content"] for doc in documents_data[:size]
                ]
                
                # Create document objects
                documents = []
                for i, doc_text in enumerate(docs_content):
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "source": f"hf_doc_{i}",
                            "doc_id": i
                        }
                    )
                    documents.append(doc)
                
                # We need to generate QA pairs since this dataset doesn't have them
                st.info("No QA pairs found in the dataset. Will need to generate synthetic QA pairs.")
                
                dataset = {
                    "documents": docs_content
                }
                
                return dataset, documents
                
            except Exception as e2:
                st.error(f"Error loading documentation dataset: {e2}")
                
                # Create fallback dataset
                st.info("Creating a fallback dataset...")
                
                # Fallback data
                fallback_docs = [
                    "The Hugging Face Datasets library provides a simple and efficient way to load, process, and share datasets. You can work with datasets of any size, from small datasets that fit in memory to datasets with millions or billions of elements.",
                    "Transformers provides APIs to download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you time from training a model from scratch.",
                    "The Hugging Face Hub is a platform where you can share, explore, and collaborate on machine learning models, datasets, and demos. It hosts over 350,000 models, 75,000 datasets, and 150,000 demo apps (Spaces), all open source and publicly available.",
                    "Tokenizers provides implementations of today's most used tokenizers that are optimized for both research and production. The library supports all the necessary steps to convert text into numbers, such as tokenization, normalization, and vocabulary management.",
                    "Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code. It simplifies the process of writing code that can run on multiple GPUs or TPUs.",
                    "Optimum provides a simple interface to optimize and run models on different hardware, including GPUs, CPUs, and specialized neural processing units (NPUs). It aims to make it easy to train and run models on various devices.",
                ]
                
                # Create Document objects
                documents = []
                for i, doc_text in enumerate(fallback_docs):
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "source": f"fallback_doc_{i}",
                            "doc_id": i
                        }
                    )
                    documents.append(doc)
                
                dataset = {
                    "documents": fallback_docs
                }
                
                return dataset, documents
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None

# Function to generate synthetic QA pairs
def generate_synthetic_qa_pairs(documents, llm):
    # Generation prompt for synthetic QA pairs
    qa_gen_prompt = """
    Based on the following document, generate 2-3 question-answer pairs that can be used to evaluate a RAG system.
    
    Document:
    {document}
    
    Generate pairs in the following format:
    Q1: [Question 1]
    A1: [Answer 1]
    
    Q2: [Question 2]
    A2: [Answer 2]
    
    Q3: [Question 3]
    A3: [Answer 3]
    
    The questions should be diverse and test different aspects of the document.
    The answers should be comprehensive and based solely on the information in the document.
    """
    
    qa_gen_chain = (
        PromptTemplate.from_template(qa_gen_prompt) 
        | llm 
        | StrOutputParser()
    )
    
    # Generate QA pairs for each document
    synthetic_questions = []
    synthetic_answers = []
    synthetic_contexts = []
    
    # Limit to a reasonable number of documents
    docs_to_process = min(10, len(documents))
    
    with st.status("Generating QA pairs...") as status:
        for i in range(docs_to_process):
            status.update(f"Processing document {i+1}/{docs_to_process}")
            doc = documents[i]
            
            # Generate QA pairs
            qa_pairs_text = qa_gen_chain.invoke({"document": doc.page_content})
            
            # Parse the generated QA pairs
            lines = qa_pairs_text.strip().split('\n')
            
            q_indices = [i for i, line in enumerate(lines) if line.startswith('Q')]
            
            for j, q_idx in enumerate(q_indices):
                # Determine the end of this QA pair
                next_q_idx = q_indices[j+1] if j+1 < len(q_indices) else len(lines)
                
                # Extract question and answer
                q_line = lines[q_idx]
                question = q_line[q_line.find(':')+1:].strip()
                
                # Find the matching A line
                a_idx = q_idx + 1
                while a_idx < next_q_idx:
                    if lines[a_idx].startswith('A'):
                        a_line = lines[a_idx]
                        answer = a_line[a_line.find(':')+1:].strip()
                        break
                    a_idx += 1
                else:
                    # No answer found, skip this question
                    continue
                
                # Add to our collections
                synthetic_questions.append(question)
                synthetic_answers.append(answer)
                synthetic_contexts.append([doc.page_content])
    
    return synthetic_questions, synthetic_answers, synthetic_contexts

def get_vectorstore(documents):
    """Load existing vector store without recreating it"""
    from langchain_community.vectorstores import Chroma
    import chromadb
    
    # Use the same embedding model that was used to create the vector store
    # This is likely the HuggingFace model since you mentioned dimensional issues earlier
    embedding_model = get_embeddings(
        embedding_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create the Chroma client with read-only flag
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=False,
            is_persistent=True,
            readonly=True  # Set readonly to avoid write errors
        )
    )
    
    # Wrap the client with LangChain's Chroma class
    with st.spinner("Loading existing vector store..."):
        try:
            vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model,
            )
            st.success(f"Successfully loaded existing vector store: {COLLECTION_NAME}")
            return vectorstore, embedding_model
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.warning("Will continue with an empty retriever - results may not be accurate")
            
            # Return an empty retriever that won't crash but won't retrieve anything useful
            class EmptyRetriever:
                def invoke(self, query):
                    return []
            
            empty_vectorstore = type('', (), {
                'as_retriever': lambda self, **kwargs: EmptyRetriever()
            })()
            
            return empty_vectorstore, embedding_model
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
                return pickle.load(f)
        return None
    except Exception as e:
        st.warning(f"Could not load previous analysis: {e}")
        return None

# Function to save analysis results
def save_analysis_results(results):
    try:
        with open(ANALYSIS_FILE, 'wb') as f:
            pickle.dump(results, f)
        st.success("Analysis results saved!")
    except Exception as e:
        st.error(f"Could not save analysis results: {e}")

# Start app initialization
if "app_initialized" not in st.session_state:
    with st.spinner("Initializing RAG evaluation..."):
        # Load dataset
        dataset, documents = load_hf_documentation_dataset()
        
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
            
            # Generate QA pairs if needed
            if "questions" not in dataset:
                st.info("Generating synthetic QA pairs from documents...")
                st.session_state.needs_qa_pairs = True
            else:
                st.success("Initialization complete! Dataset loaded and RAG models ready.")

# Generate QA pairs if needed
if st.session_state.get("needs_qa_pairs", False) and "app_initialized" in st.session_state:
    with st.spinner("Generating synthetic QA pairs..."):
        # Get LLM
        llm = st.session_state.rag_models["llm"]
        
        # Generate QA pairs
        questions, answers, contexts = generate_synthetic_qa_pairs(
            st.session_state.documents, 
            llm
        )
        
        # Store the synthetic dataset
        st.session_state.dataset["questions"] = questions
        st.session_state.dataset["answers"] = answers
        st.session_state.dataset["contexts"] = contexts
        
        # Reset flag
        st.session_state.needs_qa_pairs = False
        
        st.success(f"Generated {len(questions)} synthetic QA pairs!")

# MAIN CONTENT AREA
if "app_initialized" in st.session_state and not st.session_state.get("needs_qa_pairs", False):
    # Analysis section at the top
    st.header("RAG Performance Analysis")
    
    # Check if we have analysis results
    if st.session_state.analysis_results:
        # Display existing analysis
        all_results = st.session_state.analysis_results
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Simple RAG")
            simple_result = next((r for r in all_results if r["name"] == "Simple RAG"), None)
            if simple_result:
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
    with st.expander("About Synthetic Dataset Generation for RAG Evaluation"):
        st.markdown("""
        Creating synthetic datasets for RAG evaluation is a powerful technique to test and improve RAG systems without relying on manually labeled data.

        The process typically involves:
        1. **Document Selection**: Choosing representative documents from your knowledge base
        2. **Question Generation**: Using an LLM to generate relevant questions based on the documents
        3. **Answer Generation**: Creating ground truth answers for each question
        4. **Quality Filtering**: Ensuring the generated QA pairs are diverse and high-quality

        This approach enables effective evaluation of RAG systems, particularly for specialized domains or when labeled data is scarce.

        For more information, check out the [RAG Evaluation Cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation).
        """)
