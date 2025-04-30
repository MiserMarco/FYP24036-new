
import os
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
import re
import json
import time
from sklearn.metrics.pairwise import cosine_similarity


client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key='sk-AKQSlxgw2L8tneJkdHbxfanF0hvkWiVlAdLBtLzUnjdqKnZX'  # Retrieve the API key from environment variables
)

# 添加读取md文件的函数
def read_md_file(file_path):
    """
    Read the content of a Markdown file.

    Args:
        file_path (str): Path to the Markdown file

    Returns:
        str: Content of the Markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple whitespace characters (including newlines and tabs) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR issues by replacing tab and newline characters with a space
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')
    
    # Remove any leading or trailing whitespace and ensure single spaces between words
    text = ' '.join(text.split())
    
    return text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        List[Dict]: List of chunks with text and metadata
    """
    chunks = []  # Initialize an empty list to store chunks
    
    # Iterate over the text with the specified chunk size and overlap
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
        if chunk:  # Ensure we don't add empty chunks
            chunk_data = {
                "text": chunk,  # The chunk text
                "metadata": {
                    "start_char": i,  # Start character index of the chunk
                    "end_char": i + len(chunk)  # End character index of the chunk
                }
            }
            chunks.append(chunk_data) 
    
    print(f"Created {len(chunks)} text chunks") 
    return chunks

def create_embeddings(texts):
    """
    Create embeddings for a list of texts using OpenAI's embedding model.

    Args:
        texts (List[str]): List of texts to embed

    Returns:
        List[np.ndarray]: List of embeddings
    """
    print("Creating embeddings...")
    embeddings = []  # Initialize an empty list to store embeddings
    for text in texts:
        # Create an embedding for the text using OpenAI's embedding model
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Specify the embedding model
            input=text  # Input text to embed
        )
        # Extract the embedding vector from the response
        embedding = response.data[0].embedding
        # Append the embedding to the list
        embeddings.append(embedding)
    print("Embeddings created")
    return embeddings
class SimpleVectorStore:
    """
    A simple in-memory vector store for storing text and their corresponding embeddings.
    """
    def __init__(self):
        """
        Initialize the vector store with empty lists for texts, embeddings, and metadata.
        """
        self.texts = []  # List to store text content
        self.vectors = []  # List to store embedding vectors
        self.metadata = []  # List to store metadata for each text
    def add_item(self, text, embedding, metadata):
        """
        Add a single item to the vector store.
        Args:
            text (str): Text content
            embedding (List[float]): Embedding vector
            metadata (Dict): Metadata associated with the text
        """
        self.texts.append(text)  # Append text to the list
        self.vectors.append(embedding)  # Append embedding vector to the list
        self.metadata.append(metadata)  # Append metadata to the list
    def add_items(self, items, embeddings):
        """
        Add multiple items to the vector store.
        Args:
            items (List[Dict]): List of text items
            embeddings (List[List[float]]): List of embedding vectors
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(item, embedding, {"id": i})  # Add each item with its embedding and metadata
    def query(self, query_embedding, top_k=5):
        """
        Query the vector store for the top-k most similar items.
        Args:
            query_embedding (List[float]): Query embedding vector
            top_k (int): Number of top results to return
        Returns:
            List[Tuple[Dict, float]]: List of tuples containing the item and its similarity score
        """
        # Compute cosine similarity between the query embedding and all stored embeddings
        similarities = cosine_similarity([query_embedding], self.vectors)[0]
        # Combine similarities with items and metadata
        results = [(item, similarity) for item, similarity in zip(self.metadata, similarities)]
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        # Return the top-k results
        return results[:top_k]
def create_bm25_index(chunks):
    """
    Create a BM25 index from a list of chunks.
    Args:
        chunks (List[Dict]): List of text chunks
    Returns:
        BM25Okapi: BM25 index object
    """
    # Extract the text content from each chunk
    chunk_texts = [chunk["text"] for chunk in chunks]
    # Tokenize the chunk texts into individual tokens
    tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
    # Create a BM25 index from the tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def bm25_search(bm25, chunks, query, k=5):
    """
    Search the BM25 index with a query.
    
    Args:
        bm25 (BM25Okapi): BM25 index
        chunks (List[Dict]): List of text chunks
        query (str): Query string
        k (int): Number of results to return
        
    Returns:
        List[Dict]: Top k results with scores
    """
    # Tokenize the query by splitting it into individual words
    query_tokens = query.split()
    
    # Get BM25 scores for the query tokens against the indexed documents
    scores = bm25.get_scores(query_tokens)
    
    # Initialize an empty list to store results with their scores
    results = []
    
    # Iterate over the scores and corresponding chunks
    for i, score in enumerate(scores):
        # Create a copy of the metadata to avoid modifying the original
        metadata = chunks[i].get("metadata", {}).copy()
        # Add index to metadata
        metadata["index"] = i
        
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # Add metadata with index
            "bm25_score": float(score)
        })
    
    # Sort the results by BM25 score in descending order
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    # Return the top k results
    return results[:k]

def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    Perform fusion retrieval combining vector-based and BM25 search.
    
    Args:
        query (str): Query string
        chunks (List[Dict]): Original text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of results to return
        alpha (float): Weight for vector scores (0-1), where 1-alpha is BM25 weight
        
    Returns:
        List[Dict]: Top k results based on combined scores
    """
    print(f"Performing fusion retrieval for query: {query}")
    
    # Define small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Get vector search results
    query_embedding = create_embeddings(query)  # Create embedding for the query
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # Perform vector search
    
    # Get BM25 search results
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # Perform BM25 search
    
    # Create dictionaries to map document index to score
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    # Ensure all documents have scores for both methods
    all_docs = vector_store.get_all_documents()
    combined_results = []
    
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)  # Get vector score or 0 if not found
        bm25_score = bm25_scores_dict.get(i, 0.0)  # Get BM25 score or 0 if not found
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    # Extract scores as arrays
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # Normalize scores
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    
    # Compute combined scores
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    # Add combined scores to results
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    # Sort by combined score (descending)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Return top k results
    top_results = combined_results[:k]
    
    print(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results

# 定义一个新的函数来综合处理文档
def process_document(md_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for fusion retrieval.
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
    Returns:
        Tuple[List[Dict], SimpleVectorStore, BM25Okapi]: Chunks, vector store, and BM25 index
    """
    # Extract text from the MD file
    text = read_md_file(md_path)
    # Clean the extracted text to remove extra whitespace and special characters
    cleaned_text = clean_text(text)
    # Split the cleaned text into overlapping chunks
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
    # Extract the text content from each chunk for embedding creation
    chunk_texts = [chunk["text"] for chunk in chunks]
    print("Creating embeddings for chunks...")
    # Create embeddings for the chunk texts
    embeddings = create_embeddings(chunk_texts)
    # Initialize the vector store
    vector_store = SimpleVectorStore()
    # Add the chunks and their embeddings to the vector store
    vector_store.add_items(chunks, embeddings)
    print(f"Added {len(chunks)} items to vector store")
    # Create a BM25 index from the chunks
    bm25_index = create_bm25_index(chunks)
    # Return the chunks, vector store, and BM25 index
    return chunks, vector_store, bm25_index

#  Generate a response based on the query and context.
def generate_response(query, context):
    """
    Generate a response based on the query and context.
    
    Args:
        query (str): User query
        context (str): Context from retrieved documents
        
    Returns:
        str: Generated response
    """
    # Define the system prompt to guide the AI assistant
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    # Format the user prompt with the context and query
    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    # Generate the response using the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # Specify the model to use
        messages=[
            {"role": "system", "content": system_prompt},  # System message to guide the assistant
            {"role": "user", "content": user_prompt}  # User message with context and query
        ],
        temperature=0.1  # Set the temperature for response generation
    )
    
    # Return the generated response
    return response.choices[0].message.content

# 主函数，处理所有文档并进行检索
def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    Answer a query using fusion RAG.
    
    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        alpha (float): Weight for vector scores
        
    Returns:
        Dict: Query results including retrieved documents and response
    """
    # Retrieve documents using fusion retrieval method
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)
    
    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)
    
    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def vector_only_rag(query, vector_store, k=5):
    """
    Answer a query using only vector-based RAG.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to retrieve
        
    Returns:
        Dict: Query results
    """
    # Create query embedding
    query_embedding = create_embeddings(query)
    
    # Retrieve documents using vector-based similarity search
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)
    
    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)
    
    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def bm25_only_rag(query, chunks, bm25_index, k=5):
    """
    Answer a query using only BM25-based RAG.
    
    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        
    Returns:
        Dict: Query results
    """
    # Retrieve documents using BM25 search
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    
    # Format the context from the retrieved documents by joining their text with separators
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    
    # Generate a response based on the query and the formatted context
    response = generate_response(query, context)
    
    # Return the query, retrieved documents, and the generated response
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
    """
    Compare different retrieval methods for a query.
    
    Args:
        query (str): User query
        chunks (List[Dict]): Text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        alpha (float): Weight for vector scores in fusion retrieval
        reference_answer (str, optional): Reference answer for comparison
        
    Returns:
        Dict: Comparison results
    """
    print(f"\n=== Comparing retrieval methods for query: {query} ===\n")
    
    # Run vector-only RAG
    print("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(query, vector_store, k)
    
    # Run BM25-only RAG
    print("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)
    
    # Run fusion RAG
    print("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)
    
    # Compare responses from different retrieval methods
    print("\nComparing responses...")
    comparison = evaluate_responses(
        query, 
        vector_result["response"], 
        bm25_result["response"], 
        fusion_result["response"],
        reference_answer
    )
    
    # Return the comparison results
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }