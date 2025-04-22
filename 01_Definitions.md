---
jupyter:
  jupytext:
    formats: md:markdown
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from IPython.display import HTML

# Defining custom styling for our notebook components
HTML('''
<style>
/* Global styles */
body {
    font-family: 'Crimson Pro', 'Palatino', 'Georgia', serif;
    line-height: 1.6;
    color: #3c3c3c;
    background-color: #fcfbf8;
}

h1, h2, h3, h4 {
    font-family: 'Quattrocento Sans', 'Avenir', 'Helvetica Neue', sans-serif;
    color: #545454;
}

/* Custom component styles */
.definition-block {
    background-color: #e8e2d6;
    border-left: 5px solid #ad8e70;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 20px 0;
}

.explanation-block {
    background-color: #e5edd0;
    border-left: 5px solid #789259;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 20px 0;
}

.resources-block {
    background-color: #d6e2e8;
    border-left: 5px solid #6a8eaa;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 20px 0;
}

.outcome-block {
    background-color: #f0e6d7;
    border: 2px solid #c0a080;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 30px 0;
    text-align: center;
    font-size: 1.1em;
}

.image-container {
    text-align: center;
    margin: 20px 0;
}

.grid-container {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin: 20px 0;
}

.grid-item {
    background-color: #f7f5f0;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

code {
    background-color: #f0ebe2;
    color: #734f3e;
    padding: 2px 5px;
    border-radius: 4px;
}
</style>
''')
```

# MIND CRAFT: Building Intelligent Agents with Memory & Knowledge

Welcome to the first section of our crash course on developing intelligent agent systems. This notebook is your guide to understanding and implementing the fundamental building blocks of modern AI agents.

## What You'll Learn in This Crash Course

This crash course will take you on a journey from understanding the core concepts of knowledge, memory, and agency in AI systems to implementing sophisticated agentic architectures using LangGraph. We'll explore how these components interact to create systems that can reason, remember, and act autonomously to achieve complex goals.

<div class="outcome-block">
By the end of this course, you'll be able to design, implement, and evaluate advanced AI agent systems that effectively utilize knowledge retrieval and memory to solve real-world problems.
</div>

```python
# Quick setup to ensure our custom blocks display correctly in the notebook
from IPython.display import Markdown, display

def show_definition(text):
    display(Markdown(f'<div class="definition-block">{text}</div>'))
    
def show_explanation(text):
    display(Markdown(f'<div class="explanation-block">{text}</div>'))
    
def show_resources(text):
    display(Markdown(f'<div class="resources-block">{text}</div>'))

def show_grid(items, columns=2):
    html = '<div class="grid-container" style="grid-template-columns: repeat(' + str(columns) + ', 1fr);">'
    for item in items:
        html += f'<div class="grid-item">{item}</div>'
    html += '</div>'
    display(HTML(html))
```

## 1. Definitions: Knowledge, Memory, and Agents

Before diving into implementation details, we need to establish a clear understanding of the fundamental concepts that form the foundation of our course.

### Introduction

The field of generative AI has evolved rapidly from simple language models to sophisticated agentic systems. This evolution has been marked by increasing capabilities in knowledge retrieval, memory utilization, and autonomous decision-making. Understanding these core concepts is essential for building effective AI systems that can assist with complex tasks.

Let's begin by exploring the fundamental definitions and relationships between knowledge, memory, and agents in AI systems.

### Knowledge in AI Systems

```python
show_definition("<strong>Knowledge</strong>: Information that AI systems can access and utilize to perform tasks, answer questions, and make decisions. It represents the \"what\" of AI understanding - facts, concepts, and relationships that the system knows about the world.")
```

#### Types of Knowledge in AI

AI systems access knowledge through two primary mechanisms:

```python
show_grid([
    """<h4>Parametric Knowledge</h4>
    Information encoded directly within the model's weights during training. This knowledge is "baked into" the model itself.""",
    
    """<h4>Non-Parametric Knowledge</h4>
    Information stored outside the model's parameters, typically in external databases or knowledge bases that can be queried at runtime."""
])
```

##### Parametric Knowledge

```python
show_explanation("""<strong>Characteristics of Parametric Knowledge:</strong>
<ul>
<li>Stored within the model's parameters (weights and biases)</li>
<li>Fixed after training (without fine-tuning or retraining)</li>
<li>Fast to access, as it requires no external lookup</li>
<li>May become outdated as the world changes</li>
<li>Limited by model size and training data</li>
</ul>""")
```

For example, when a large language model (LLM) knows that "Paris is the capital of France" without looking it up externally, it's using parametric knowledge stored in its weights.

##### Non-Parametric Knowledge

```python
show_explanation("""<strong>Characteristics of Non-Parametric Knowledge:</strong>
<ul>
<li>Stored in external systems (databases, knowledge graphs, documents)</li>
<li>Can be updated without retraining the model</li>
<li>Requires retrieval mechanisms (search, lookup)</li>
<li>Easily expandable without changing the model itself</li>
<li>May increase latency due to retrieval time</li>
</ul>""")
```

When an AI system looks up current weather data or retrieves information from a company's database, it's utilizing non-parametric knowledge.

#### Knowledge Representation Methods

How knowledge is represented significantly impacts how AI systems can use it. Common knowledge representation methods include:

```python
show_grid([
    """<h4>Vector Embeddings</h4>
    Represent concepts, words, or documents as numerical vectors in a high-dimensional space, where semantic similarity is captured by vector proximity.""",
    
    """<h4>Knowledge Graphs</h4>
    Structured representations of knowledge as a network of entities (nodes) and relationships (edges) between them.""",
    
    """<h4>Symbolic Representations</h4>
    Formal logical structures that enable reasoning through rules and relationships, often using predicate logic.""",
    
    """<h4>Distributed Representations</h4>
    Information distributed across many parameters, as in neural networks, where knowledge is represented implicitly across connections."""
], columns=2)
```

##### Vector Embeddings Example

```python
# Example: Creating word embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["Paris is the capital of France", "Berlin is the capital of Germany"]
embeddings = model.encode(sentences)
print(f"Shape of embeddings: {embeddings.shape}")
```

##### Knowledge Graph Example

```python
# Example: Simple knowledge graph representation
graph = {
    "Paris": {"is_capital_of": "France", "population": "2.2 million"},
    "France": {"has_capital": "Paris", "continent": "Europe"}
}
print("Sample knowledge graph:")
import json
print(json.dumps(graph, indent=2))
```

```python
show_resources("""<strong>Key Resources on Knowledge in AI:</strong>
<ul>
<li><a href="https://medium.com/towards-data-science/knowledge-retrieval-takes-center-stage-183be733c6e8" target="_blank">Gadi Singer's "Knowledge Retrieval Takes Center Stage"</a> explores the shift from parametric to retrieval-centric models</li>
<li><a href="https://medium.com/towards-data-science/seat-of-knowledge-ai-systems-with-deeply-structure-knowledge-37f1a5ab4bc5" target="_blank">"Seat of Knowledge: AI Systems with Deeply Structured Knowledge"</a> provides a classification of AI systems based on knowledge representation</li>
<li><a href="https://www.ibm.com/think/topics/agentic-rag" target="_blank">IBM's research on knowledge integration in AI systems</a> offers practical approaches to combining parametric and non-parametric knowledge</li>
</ul>""")
```

### Memory in AI Agents

```python
show_definition("<strong>Memory</strong>: An AI system's ability to store and recall past experiences, interactions, or computational states to maintain context and influence future behavior. Memory enables persistence and continuity in AI systems.")
```

#### Differentiating Memory from Knowledge

While knowledge and memory are related concepts, they serve different functions in AI systems:

```python
show_explanation("""<strong>Knowledge vs. Memory:</strong>
<ul>
<li><strong>Knowledge</strong> is about factual information and understanding of concepts (the "what")</li>
<li><strong>Memory</strong> is about recording experiences and context over time (the "when" and "how")</li>
<li><strong>Knowledge</strong> might tell an AI about calendar systems and how dates work</li>
<li><strong>Memory</strong> helps an AI remember that the user scheduled a meeting last Tuesday</li>
</ul>""")
```

This distinction becomes particularly important when designing AI systems that need to maintain context across multiple interactions with users.

#### Types of Memory in AI Systems

AI systems implement several forms of memory, inspired by human cognitive systems:

```python
show_grid([
    """<h4>Short-term Memory (Working Memory)</h4>
    Temporary storage for ongoing conversations or tasks, typically limited in scope and duration.""",
    
    """<h4>Episodic Memory</h4>
    Records specific past experiences and interactions, similar to how humans remember events.""",
    
    """<h4>Semantic Memory</h4>
    Stores structured factual knowledge, definitions, and rules about specific domains.""",
    
    """<h4>Procedural Memory</h4>
    Contains information about how to perform tasks or follow processes."""
], columns=2)
```

##### Implementation Approaches

```python
show_grid([
    """<h5>Short-term Memory</h5>
    <ul>
    <li>Storing recent conversation turns in context window</li>
    <li>Maintaining state within a session</li>
    <li>Tracking active goals and subtasks</li>
    </ul>""",
    
    """<h5>Episodic Memory</h5>
    <ul>
    <li>Conversation logs with timestamps</li>
    <li>Interaction history databases</li>
    <li>Vector databases of past exchanges</li>
    </ul>""",
    
    """<h5>Semantic Memory</h5>
    <ul>
    <li>Knowledge graphs of user preferences</li>
    <li>Structured databases of learned facts</li>
    <li>Vector stores of semantic information</li>
    </ul>""",
    
    """<h5>Procedural Memory</h5>
    <ul>
    <li>Stored procedures or functions</li>
    <li>Task templates and workflows</li>
    <li>Learned patterns for multi-step processes</li>
    </ul>"""
], columns=2)
```

#### Importance of Memory for AI Agents

Memory capabilities are essential for creating persistent, context-aware AI agents:

```python
show_grid([
    """<strong>Continuity</strong><br>Enables coherent multi-turn interactions""",
    """<strong>Personalization</strong><br>Allows agents to adapt to individual users over time""",
    """<strong>Learning</strong><br>Provides experiences from which to improve performance""",
    """<strong>Relationship Building</strong><br>Creates a sense of persistence and familiarity""",
    """<strong>Complex Task Management</strong><br>Supports handling of multi-stage processes"""
], columns=3)
```

```python
show_resources("""<strong>Key Resources on Memory in AI:</strong>
<ul>
<li><a href="https://www.ibm.com/think/topics/ai-agent-memory" target="_blank">IBM's comprehensive guide on "What Is AI Agent Memory?"</a> explores different memory types and implementations</li>
<li><a href="https://blog.langchain.dev/memory-for-agents/" target="_blank">"Memory for agents" from the LangChain Blog</a> discusses practical memory implementations for agent systems</li>
<li><a href="https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents" target="_blank">"Memory: The secret sauce of AI agents" from Decoding ML</a> details how different memory types enhance agent capabilities</li>
</ul>""")
```

### Agents in Generative AI

```python
show_definition("<strong>Agents</strong>: Autonomous systems that perceive their environment, make decisions, and take actions to achieve specific goals. AI agents use generative models as their reasoning engine while incorporating additional capabilities for interaction and task completion.")
```

#### Components of AI Agents

Modern AI agents typically consist of several interdependent components:

```python
show_grid([
    """<h4>1. Perception</h4>
    Mechanisms for receiving and processing input from the environment or users.<br><br>
    <strong>Examples:</strong> Natural language understanding, vision systems, data connectors""",
    
    """<h4>2. Reasoning</h4>
    Cognitive capabilities for understanding, planning, and decision-making.<br><br>
    <strong>Examples:</strong> LLMs for reasoning, planning systems, domain-specific models""",
    
    """<h4>3. Memory</h4>
    Systems for storing and retrieving past experiences and knowledge.<br><br>
    <strong>Examples:</strong> Short-term context, long-term storage, episodic records""",
    
    """<h4>4. Action</h4>
    Capabilities for executing decisions and interacting with the environment.<br><br>
    <strong>Examples:</strong> Tool usage (APIs), text generation, database operations"""
], columns=2)
```

#### Evolution from Basic LLMs to Agentic Systems

The progression from simple language models to agentic systems represents a significant evolution in AI capabilities:

```python
show_grid([
    """<h4>Stage 1: Basic Language Models</h4>
    Generate text based on prompts without agency or persistence.
    <pre>
    User: "What is the capital of France?"
    LLM: "The capital of France is Paris."
    </pre>""",
    
    """<h4>Stage 2: Conversational AI</h4>
    Maintain context within a conversation but limited to textual interaction.
    <pre>
    User: "What is the capital of France?"
    AI: "The capital of France is Paris."
    User: "What is its population?"
    AI: "Paris has a population of approximately 2.2 million people."
    </pre>""",
    
    """<h4>Stage 3: Tool-Using AI</h4>
    Capable of using external tools and APIs to accomplish tasks beyond text generation.
    <pre>
    User: "What's the weather in Tokyo right now?"
    AI: [Calls weather API] "Currently in Tokyo, it's 22°C (72°F) and partly cloudy."
    </pre>""",
    
    """<h4>Stage 4: Agentic Systems</h4>
    Autonomous systems that can plan, reason, and execute complex multi-step tasks with minimal supervision.
    <pre>
    User: "Plan my trip to Paris next month."
    Agent: [Plans itinerary, checks flights, books accommodations, creates cohesive plan across multiple tools and data sources]
    </pre>"""
], columns=2)
```

```python
show_resources("""<strong>Key Resources on AI Agents:</strong>
<ul>
<li><a href="https://www.ibm.com/think/topics/agentic-reasoning" target="_blank">IBM's "What Is Agentic Reasoning?"</a> provides insights into the decision-making capabilities of agents</li>
<li><a href="https://ajithp.com/2025/04/05/llm-based-intelligent-agents/" target="_blank">"LLM-Based Intelligent Agents: Architecture and Evolution"</a> offers a comprehensive look at agent architecture</li>
<li><a href="https://www.pryon.com/landing/agentic-ai-101-how-to-unlock-the-power-of-ai-agents" target="_blank">"Agentic AI 101" from Pryon</a> explores the components and capabilities of agentic systems</li>
</ul>""")
```

### Relationships Between Knowledge, Memory, and Agency

These three concepts are deeply interconnected in advanced AI systems:

```python
show_grid([
    """<strong>Knowledge</strong> provides the information foundation upon which agents operate""",
    """<strong>Memory</strong> creates persistence and learning capabilities across interactions""",
    """<strong>Agency</strong> enables autonomous goal-directed behavior using knowledge and memory"""
], columns=3)
```

Together, they form the foundation for building sophisticated AI systems that can assist with complex tasks, adapt to user needs, and operate with increasing levels of autonomy.

### Summary

- **Knowledge** encompasses the information AI systems can access and utilize, whether stored in model parameters or external sources
- **Memory** enables AI systems to maintain context and learn from experiences over time through various storage mechanisms
- **Agents** are autonomous systems that combine perception, reasoning, memory, and action capabilities to accomplish goals
- These three components form the foundation of advanced AI systems, with each playing a critical role in overall system capabilities

```python
show_resources("""<strong>Further Reading:</strong>
<ul>
<li><a href="https://diamantai.substack.com/p/building-an-ai-agent-with-memory" target="_blank">"Building an AI Agent with Memory and Adaptability" - DiamantAI</a></li>
<li><a href="https://www.pryon.com/landing/agentic-ai-101-how-to-unlock-the-power-of-ai-agents" target="_blank">"Agentic AI 101: How to Unlock the Power of AI Agents" - Pryon</a></li>
<li><a href="https://odsc.medium.com/rag-in-2024-the-evolution-of-ai-powered-knowledge-retrieval-6d273b822c14" target="_blank">"RAG in 2024: The Evolution of AI-Powered Knowledge Retrieval"</a></li>
<li><a href="https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent/" target="_blank">"A Long-Term Memory Agent" - LangChain Documentation</a></li>
</ul>""")
```

## Next Steps

In the following sections, we'll explore these concepts in greater depth, focusing on implementation approaches for knowledge retrieval systems, memory architectures, and agentic frameworks using LangGraph and related technologies.
