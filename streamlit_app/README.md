# RAG Evaluation and Optimization Dashboard

This Streamlit application demonstrates Retrieval-Augmented Generation (RAG) implementation, evaluation, and optimization with worked examples on different datasets.

## Features

- Implementation of three RAG architectures:
  - Simple RAG (baseline)
  - Corrective RAG (CRAG)
  - Adaptive RAG

- Evaluation using RAGAS metrics:
  - Faithfulness
  - Context Precision
  - Context Recall
  - Answer Relevancy
  - Answer Correctness

- Two worked examples:
  1. **Amnesty QA Dataset**: Human rights Q&A dataset from Amnesty International reports
  2. **HuggingFace Documentation**: Synthetic dataset created from HuggingFace documentation

- Visualization of evaluation results through:
  - Metric comparison charts
  - Radar charts
  - Overall performance scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Knowledge-Memory-Agents.git
cd Knowledge-Memory-Agents/streamlit_app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate to the provided URL (typically http://localhost:8501)

3. Follow the instructions in the app to:
   - Load datasets
   - Create or load vector stores
   - Initialize RAG models
   - Test and evaluate the different RAG architectures

## Project Structure

```
streamlit_app/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── pages/                  # Streamlit pages for each example
│   ├── 1_Amnesty_QA_Example.py
│   └── 2_HuggingFace_Doc_Example.py
├── components/             # Reusable UI components
│   └── chat_interface.py
├── services/               # RAG implementations
│   ├── simple_rag.py
│   ├── crag.py
│   └── adaptive_rag.py
├── utils/                  # Utility functions
│   ├── embedding_utils.py
│   ├── llm_utils.py
│   ├── vectorstore_utils.py
│   └── evaluation_utils.py
└── data/                   # Directory for storing data
```

## Datasets

### Amnesty QA Dataset
- Source: [explodinggradients/amnesty_qa](https://huggingface.co/datasets/explodinggradients/amnesty_qa)
- Description: Q&A dataset created from Amnesty International research reports

### HuggingFace Documentation Dataset
- Source: [m-ric/huggingface_doc](https://huggingface.co/datasets/m-ric/huggingface_doc) and [m-ric/huggingface_doc_qa_eval](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval)
- Description: Documentation content and synthetic QA pairs based on HuggingFace documentation

## Dependencies

- streamlit
- langchain
- chroma
- openai
- pandas
- matplotlib
- seaborn
- ragas
- and more (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [RAGAS](https://github.com/explodinggradients/ragas) for the evaluation metrics
- [HuggingFace](https://huggingface.co/) for the datasets
- [LangChain](https://github.com/langchain-ai/langchain) for the RAG components
