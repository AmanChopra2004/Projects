# Knowledge Graph-based Question Answering System

A sophisticated question answering system that combines graph databases, language models, and vector search to provide accurate answers based on structured and unstructured data.

## Features

- Entity extraction and recognition
- Graph-based knowledge representation
- Hybrid search combining structured and unstructured data
- Conversation history support
- Natural language answer generation
- Interactive graph visualization
- Fuzzy text search capabilities

## Architecture

The system consists of several key components:

1. **Graph Database**: Neo4j for storing structured knowledge
2. **Vector Store**: For similarity-based unstructured text search
3. **Language Model**: Groq's Mixtral-8x7b-32768 for natural language processing
4. **Graph Visualization**: yFiles for Jupyter graph visualization

## Prerequisites

- Python 3.x
- Neo4j database instance
- Groq API key
- Required Python packages (see Installation section)

## Installation

1. Install required packages:
```bash
pip install --upgrade --quiet langchain langchain-community langchain-groq langchain-experimental neo4j wikipedia tiktoken yfiles_jupyter_graphs sentence-transformers
```

2. Set up environment variables:
```python
NEO4J_URI="your_neo4j_uri"
NEO4J_USERNAME="your_username"
NEO4J_PASSWORD="your_password"
GROQ_API_KEY="your_groq_api_key"
```

## System Components

### 1. Entity Extraction
- Uses LLM to identify and extract entities from text
- Supports person, organization, and business entities

### 2. Graph Transformation
- Converts text documents into graph structures
- Creates nodes and relationships for knowledge representation

### 3. Search System
- Structured retrieval using Neo4j fulltext search
- Unstructured retrieval using vector similarity search
- Fuzzy matching for improved search results

### 4. Question Answering Pipeline
- Question analysis and entity extraction
- Context retrieval from both structured and unstructured sources
- Natural language answer generation

### 5. Visualization
- Interactive graph visualization using yFiles
- Node and relationship display
- Custom styling and layout options

## Usage

1. Initialize the system:
```python
from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph()
```

2. Load and process documents:
```python
from langchain.document_loaders import WikipediaLoader
raw_documents = WikipediaLoader(query="your_topic").load()
```

3. Transform documents into graph format:
```python
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)
```

4. Ask questions:
```python
response = chain.invoke("Your question here")
print(response)
```

## Example Queries

The system can handle various types of questions:
- Factual queries about entities
- Relationship questions
- Complex queries requiring multiple data sources
- Follow-up questions using conversation history

## Visualization Example

```python
showGraph("MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50")
```

## Performance Optimization

- Fulltext indexing for fast entity search
- Efficient graph traversal patterns
- Hybrid search combining multiple data sources
- Fuzzy matching for improved search accuracy

## Limitations

- Requires active Neo4j database connection
- API key dependencies for external services
- Limited by the quality and coverage of input data
- Processing time may vary with data size

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request



## Acknowledgments

- LangChain for the core framework
- Neo4j for graph database capabilities
- Groq for language model support
- yFiles for graph visualization
