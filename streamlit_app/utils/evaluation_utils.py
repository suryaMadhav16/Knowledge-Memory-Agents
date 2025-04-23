"""
Evaluation utilities for RAG applications.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ragas.metrics import (
    Faithfulness,
    ContextRelevance,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    AnswerCorrectness
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from datasets import Dataset

def prepare_evaluation_dataset(
    questions: List[str],
    answers: List[str],
    references: List[str],
    contexts: List[List[str]]
) -> EvaluationDataset:
    """
    Prepare an evaluation dataset for RAGAS.
    
    Args:
        questions: List of questions
        answers: List of generated answers
        references: List of reference answers
        contexts: List of retrieved contexts for each question
        
    Returns:
        RAGAS EvaluationDataset
    """
    # Create a dictionary with the required fields
    data = {
        "user_input": questions,
        "response": answers,
        "reference": references,
        "retrieved_contexts": contexts
    }
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Create Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    # Convert to RAGAS EvaluationDataset
    return EvaluationDataset.from_huggingface(hf_dataset)

def evaluate_rag_system(
    evaluation_dataset: EvaluationDataset,
    llm_evaluator,
    metrics: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate a RAG system using RAGAS metrics.
    
    Args:
        evaluation_dataset: The dataset to evaluate
        llm_evaluator: The LLM evaluator to use
        metrics: List of RAGAS metrics to use (if None, uses default metrics)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Use default metrics if none provided
    if metrics is None:
        metrics = [
            Faithfulness(llm=llm_evaluator),
            ContextPrecision(llm=llm_evaluator),
            ContextRecall(llm=llm_evaluator),
            AnswerRelevancy(llm=llm_evaluator),
            AnswerCorrectness(llm=llm_evaluator)
        ]
    
    # Run evaluation
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm_evaluator
    )
    
    return {
        "name": "RAG System",
        "results": results._repr_dict,
        "details": results
    }

def create_metric_comparison_chart(results_list: List[Dict[str, Any]]) -> plt.Figure:
    """
    Create a bar chart comparing metrics across systems.
    
    Args:
        results_list: List of evaluation results
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    systems = [r["name"] for r in results_list]
    metrics = list(results_list[0]["results"].keys())
    
    # Create a DataFrame for plotting
    plot_data = []
    for system_result in results_list:
        for metric, value in system_result["results"].items():
            plot_data.append({
                "System": system_result["name"],
                "Metric": metric,
                "Score": value
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create the bar plot
    chart = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="System",
        palette="viridis",
        ax=ax
    )
    
    # Add labels and title
    plt.title("RAG System Performance Comparison", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.0)  # Consistent y-axis limit
    plt.legend(title="RAG System", fontsize=10)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    # Add score values on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            xytext=(0, 5),
            textcoords="offset points"
        )
    
    plt.tight_layout()
    
    return fig

def create_radar_chart(results_list: List[Dict[str, Any]]) -> plt.Figure:
    """
    Create a radar chart showing the strengths of each system.
    
    Args:
        results_list: List of evaluation results
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    systems = [r["name"] for r in results_list]
    metrics = list(results_list[0]["results"].keys())
    
    # Convert metric names to more readable format
    readable_metrics = [metric.replace("_", " ").title() for metric in metrics]
    
    # Convert to scores array
    scores = np.zeros((len(systems), len(metrics)))
    for i, system_result in enumerate(results_list):
        for j, metric in enumerate(metrics):
            scores[i, j] = system_result["results"][metric]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Add the first angle again at the end to close the loop
    angles = np.append(angles, angles[0])
    
    # Add the first metric name again to close the labels
    readable_metrics.append(readable_metrics[0])
    
    # Plot each system with loop closure
    for i, system in enumerate(systems):
        values = scores[i].tolist()
        # Add the first value again to close the loop
        values.append(values[0])
        ax.plot(angles, values, linewidth=2, label=system)
        ax.fill(angles, values, alpha=0.1)
    
    # Add labels and configure the chart
    ax.set_thetagrids(np.degrees(angles[:-1]), readable_metrics[:-1])
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)  # Move radial labels to better position
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 0.1))
    
    plt.title("RAG Systems Performance Comparison", size=15)
    plt.tight_layout()
    
    return fig

def calculate_overall_scores(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate weighted average scores for each system.
    
    Args:
        results_list: List of evaluation results
        
    Returns:
        DataFrame with overall scores
    """
    # Define metric weights
    weights = {
        "faithfulness": 0.3,
        "context_precision": 0.2,
        "context_recall": 0.15,
        "answer_relevancy": 0.2,
        "answer_correctness": 0.15
    }
    
    overall_scores = []
    for result in results_list:
        weighted_sum = 0
        for metric, score in result["results"].items():
            # Match metric name to weight key (handling suffix like 'mode=f1')
            for weight_key in weights:
                if weight_key in metric:
                    weighted_sum += score * weights[weight_key]
                    break
        
        overall_scores.append({
            "System": result["name"],
            "Overall Score": weighted_sum
        })
    
    # Create dataframe and sort by score
    return pd.DataFrame(overall_scores).sort_values("Overall Score", ascending=False)
