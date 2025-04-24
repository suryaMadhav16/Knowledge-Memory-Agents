# Advanced RAG Architectures: Quiz

Student Name: Sai Surya Madhav Rebbapragada
NEU ID: 
Topic: Advanced Retrieval-Augmented Generation (RAG) Architectures


## Overview
This quiz covers key concepts from the Advanced RAG Architectures notebook, focusing on different RAG implementations, evaluation metrics, and practical applications.

## Questions

### Question 1 [Recall]
**What is the primary purpose of Retrieval-Augmented Generation (RAG) in AI systems?**

A) To reduce the training time of language models  
B) To ground language model responses in external knowledge sources  
C) To enable language models to write their own training data  
D) To eliminate the need for large language models entirely  

**Correct Answer: B**

**Explanation:** Retrieval-Augmented Generation (RAG) enhances language models by connecting them to external knowledge sources, allowing them to retrieve relevant information before generating responses. This helps ground their answers in factual information, reducing hallucinations and improving accuracy.

### Question 2 [Recall]
**Which of the following is NOT one of the core components of a standard RAG system?**

A) Document Collection  
B) Retriever  
C) Generator  
D) Retrainer  

**Correct Answer: D**

**Explanation:** Standard RAG systems typically consist of document collection (the corpus of knowledge), document processing (converting documents into a suitable format), retriever (finding relevant information based on a query), generator (usually an LLM that creates a response), and an orchestration layer. "Retrainer" is not a standard component.

### Question 3 [Recall]
**In the RAGAS evaluation framework, what does the 'Faithfulness' metric measure?**

A) How well the generated response matches the user's query  
B) How well the retrieved documents match the user's query  
C) Whether the generated response contains only information from the retrieved documents  
D) The factual accuracy of the retrieved documents  

**Correct Answer: C**

**Explanation:** Faithfulness measures whether the generated response contains only information that can be derived from the retrieved documents. It detects hallucinations and unsupported statements by calculating the ratio of claims in the response that are supported by the retrieved context to the total number of claims.

### Question 4 [Recall]
**What is the key innovation that Corrective RAG (CRAG) introduces compared to Simple RAG?**

A) It uses multiple language models instead of just one  
B) It evaluates the quality of retrieved documents before generating a response  
C) It always uses web search instead of a vector database  
D) It can generate responses without any external knowledge  

**Correct Answer: B**

**Explanation:** Corrective RAG (CRAG) introduces a self-assessment loop to evaluate the quality and relevance of retrieved documents before generation. It can grade documents, filter irrelevant content, and trigger fallback mechanisms like web search if the initially retrieved documents are insufficient or irrelevant.

### Question 5 [Recall]
**Which RAG architecture dynamically routes queries through different processing paths based on the query type?**

A) Simple RAG  
B) CRAG  
C) Adaptive RAG  
D) Self-RAG  

**Correct Answer: C**

**Explanation:** Adaptive RAG analyzes the query to determine its complexity and type, then routes it through the most appropriate processing path. This might include direct LLM generation for simple queries, standard retrieval for factual questions, or more complex paths like multi-hop retrieval for complex queries requiring information synthesis.

### Question 6 [Recall]
**What is the primary benefit of using RAGAS for evaluating RAG systems?**

A) It's faster than human evaluation  
B) It provides objective metrics across multiple dimensions of RAG performance  
C) It's integrated with all major LLM providers  
D) It automatically fixes problems in RAG implementations  

**Correct Answer: B**

**Explanation:** RAGAS provides a comprehensive suite of objective metrics that evaluate different aspects of a RAG system's performance, including faithfulness, context precision, context recall, answer relevancy, and answer correctness. This allows for systematic assessment and comparison of different RAG implementations.

### Question 7 [Application]
**In the context of RAG evaluation, what does 'Context Precision' measure?**

A) The speed at which context is retrieved  
B) The amount of context that can be processed at once  
C) The proportion of retrieved context that is relevant to the question  
D) The accuracy of the formatting of retrieved context  

**Correct Answer: C**

**Explanation:** Context Precision evaluates whether the retrieved documents are relevant to the question being asked. It measures the proportion of retrieved context that is actually useful for answering the query, with higher scores indicating more precise retrieval with less irrelevant information.

### Question 8 [Application]
**What is a key limitation of Simple RAG that more advanced architectures aim to address?**

A) It requires too much computational power  
B) It cannot handle multiple questions at once  
C) It lacks mechanisms to validate the quality of retrieved documents  
D) It only works with specific language models  

**Correct Answer: C**

**Explanation:** Simple RAG lacks mechanisms to validate the quality of retrieved documents or adapt when retrieval fails to find relevant information. Advanced architectures like CRAG and Adaptive RAG introduce components to assess document relevance, handle retrieval failures, and adapt the processing path based on query requirements.

### Question 9 [Application]
**In a RAG system, what is the primary role of the vector store?**

A) To generate text responses  
B) To store and index document embeddings for efficient similarity search  
C) To split documents into smaller chunks  
D) To translate between different languages  

**Correct Answer: B**

**Explanation:** In a RAG system, the vector store (often implemented as a vector database) stores and indexes document embeddings, enabling efficient similarity search when a query is received. This allows the system to quickly retrieve documents that are semantically similar to the query.

### Question 10 [Application]
**Which RAGAS metric would be most useful for detecting when a RAG system is making up information not present in the retrieved documents?**

A) Context Recall  
B) Answer Relevancy  
C) Faithfulness  
D) Context Precision  

**Correct Answer: C**

**Explanation:** Faithfulness specifically measures whether the generated response contains only information that can be derived from the retrieved documents. A low faithfulness score indicates that the system is introducing information (hallucinating) that isn't supported by the retrieved context.

### Question 11
**What is the purpose of the query rewriting component in Corrective RAG (CRAG)?**

A) To translate the query into multiple languages  
B) To improve search results when initial retrieval fails  
C) To make the query shorter and easier to process  
D) To correct grammatical errors in the user's query  

**Correct Answer: B**

**Explanation:** In CRAG, when initial retrieval doesn't yield relevant documents, the query may be rewritten to improve search results before triggering fallback mechanisms like web search. This helps optimize the retrieval process by reformulating the query to better match available information sources.

### Question 12 [Analysis]
**When implementing RAG with LangGraph, what is the primary advantage of using a state machine approach?**

A) It allows for loops and conditional branching in the RAG workflow  
B) It makes the system run faster  
C) It reduces the amount of code needed  
D) It eliminates the need for external knowledge sources  

**Correct Answer: A**

**Explanation:** LangGraph enables implementing RAG as a state machine, which allows for loops and conditional branching in the workflow. This is particularly useful for implementing self-reflective RAG architectures that need to make decisions based on intermediate results and potentially loop back to earlier steps.

### Question 13 [Analysis]
**What is the primary challenge addressed by using the 'Context Recall' metric in RAG evaluation?**

A) Ensuring the response is grammatically correct  
B) Verifying that all necessary information is present in the retrieved documents  
C) Checking that the system can handle multiple questions at once  
D) Measuring the speed of document retrieval  

**Correct Answer: B**

**Explanation:** Context Recall measures whether all the information needed to answer the question is present in the retrieved documents. It assesses if important information is missing from the retrieved context by calculating the proportion of claims in a reference answer that are supported by the retrieved context.

### Question 14 [Analysis]
**In document processing for RAG, what is the purpose of chunk overlap?**

A) To reduce the overall number of chunks  
B) To ensure semantic continuity across chunk boundaries  
C) To speed up the embedding process  
D) To compress documents to save storage space  

**Correct Answer: B**

**Explanation:** Chunk overlap is used in document processing to ensure semantic continuity across chunk boundaries. By allowing some content to appear in multiple adjacent chunks, important contextual information spanning the boundary between chunks is less likely to be lost during retrieval.

### Question 15 [Analysis]
**Which of the following best describes the relationship between 'Answer Relevancy' and 'Answer Correctness' in RAGAS?**

A) They are the same metric with different names  
B) Answer Relevancy measures topical alignment, while Answer Correctness measures factual accuracy  
C) Answer Relevancy is only used for open-domain questions, while Answer Correctness is for closed-domain  
D) Answer Relevancy applies to short answers, while Answer Correctness is for longer responses  

**Correct Answer: B**

**Explanation:** Answer Relevancy evaluates how well the generated response addresses the user's question regardless of factual accuracy, measuring if the answer is on-topic. Answer Correctness compares the generated answer with a reference (ground truth) answer to assess factual accuracy, combining semantic similarity with factual overlap.

### Question 16 [Analysis]
**What key capability differentiates Adaptive RAG from Simple RAG?**

A) The ability to use multiple language models  
B) The ability to generate longer responses  
C) The ability to classify queries and route them through different processing paths  
D) The ability to handle multiple languages  

**Correct Answer: C**

**Explanation:** Adaptive RAG introduces a query classifier that analyzes the query to determine its complexity and type, then routes it through the most appropriate processing path. This dynamic routing allows the system to efficiently handle different types of queries with specialized processing approaches.

### Question 17 [Analysis]
**In the context of RAG implementation with LangGraph, what is the purpose of document grading?**

A) To assign a numerical score to each document for ranking  
B) To evaluate document relevance to the query and filter irrelevant content  
C) To check documents for grammatical errors  
D) To translate documents into the query's language  

**Correct Answer: B**

**Explanation:** Document grading in RAG implementations with LangGraph involves evaluating the relevance of retrieved documents to the query. This step allows the system to filter out irrelevant content and make decisions about whether additional retrieval steps (like web search) are needed to find more relevant information.

### Question 18 [Recall]
**Which of the following is an advantage of CRAG over Simple RAG?**

A) It always produces shorter, more concise answers  
B) It can incorporate real-time web search when local retrieval is insufficient  
C) It eliminates the need for document processing  
D) It works without requiring a language model  

**Correct Answer: B**

**Explanation:** A key advantage of CRAG over Simple RAG is that it can incorporate real-time web search when local document retrieval is insufficient or irrelevant. This fallback mechanism helps ensure the system can access the most current and relevant information even when it's not available in the local knowledge base.

### Question 19 [Recall]s
**When calculating the overall RAGAS score, what approach is typically used to combine individual metrics?**

A) Only the lowest metric score is used  
B) Only the highest metric score is used  
C) An average of the component metrics is calculated  
D) The product of all metrics is calculated  

**Correct Answer: C**

**Explanation:** The combined RAGAS score is typically calculated as an average of the component scores (Faithfulness, Context Precision, Context Recall, Answer Relevancy, and Answer Correctness), sometimes with different weights assigned based on specific application requirements.

### Question 20 [Analysis]
**Which RAG architecture would be most appropriate for an application that needs to handle a wide variety of query types, from simple factual questions to complex analytical requests?**

A) Simple RAG  
B) CRAG  
C) Adaptive RAG  
D) Static RAG  

**Correct Answer: C**

**Explanation:** Adaptive RAG is designed to handle a wide variety of query types by dynamically routing them through different processing paths based on their complexity and requirements. This makes it well-suited for applications that need to efficiently handle everything from simple factual questions to complex analytical requests requiring information synthesis.
