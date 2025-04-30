import base64
import os
import json
import time
import zipfile
from pathlib import Path
import re
import uuid
import pymupdf
import gradio as gr
from gradio_pdf import PDF
from loguru import logger
from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.tools.common import do_parse, prepare_env
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_core.output_parsers import StrOutputParser
import argparse
import openai
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
api_key = 'sk-AKQSlxgw2L8tneJkdHbxfanF0hvkWiVlAdLBtLzUnjdqKnZX'
base_url = 'https://www.dmxapi.cn/v1'
client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)

# 从pdf读取text
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    print(f"Extracting text from {pdf_path}...")  # Print the path of the PDF being processed
    pdf_document = fitz.open(pdf_path)  # Open the PDF file using PyMuPDF
    text = ""  # Initialize an empty string to store the extracted text
    
    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]  # Get the page object
        text += page.get_text()  # Extract text from the page and append to the text string
    
    return text  # Return the extracted text content

# 从md读取text
def extract_text_from_md(md_path):
    """
    Extract text content from a Markdown file.

    Args:
        md_path (str): Path to the Markdown file
    Returns:
        str: Extracted text content
    """
    print(f"Extracting text from {md_path}...")  # Print the path of the Markdown file being processed
    with open(md_path, 'r', encoding='utf-8') as file:
        text = file.read()  # Read the entire content of the Markdown file
    return text  # Return the extracted text content
    


# 定义一个简单的函数用于展示上下文（这里只是简单打印，实际可根据需求完善）
def show_context(docs_content):
    """
    展示检索到的文档内容（简单打印，可根据实际需求修改）

    :param docs_content: 包含文档内容的列表
    """
    for content in docs_content:
        logger.info(content)


def read_fn(path):
    """
    从指定路径读取文件内容

    :param path: 文件路径
    :return: 文件内容
    """
    disk_rw = FileBasedDataReader(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path))


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, layout_mode, formula_enable, table_enable, language):
    """
    解析PDF文件，根据配置进行OCR识别、布局分析等操作

    :param doc_path: PDF文件路径
    :param output_dir: 输出目录
    :param end_page_id: 结束页码
    :param is_ocr: 是否进行OCR识别
    :param layout_mode: 布局模式
    :param formula_enable: 是否启用公式识别
    :param table_enable: 是否启用表格识别
    :param language: 语言
    :return: Markdown目录路径和文件名
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)
        if is_ocr:
            parse_method = "ocr"
        else:
            parse_method = "auto"
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
            f_dump_orig_pdf=False,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    :return: 0表示成功，-1表示失败
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    """
    将图像转换为base64编码的字符串

    :param image_path: 图像文件路径
    :return: base64编码的字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 将检测出来的图像，采用多模态大模型进行内容识别与分析
def image_to_text(image_path):
    """
    使用多模态大模型将图像转换为文本

    :param image_path: 图像文件路径（支持jpeg格式）
    :return: 识别的文本内容（失败返回默认提示）
    """
    """
    将图像转换为文本

    :param image_path: 图像文件路径
    :return: 识别的文本
    """
    try:
        # 读取图像文件
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # 调用OpenAI API进行图像识别
        response = client.chat.completions.create(
            model="gemini-1.5-flash-latest",  # 使用最新的模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请你对下面的图片进行内容识别与分析,如果识别为金融数据类图，请你详细分析其数值和趋势。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
                                "detail": "high"  # 可以选择"low"或"high"
                            }  
                        } 
                    ]    
                }  
            ]  
        )
        # 提取并返回识别的文本
        print('--------:',response.choices[0].message.content,'------')
        return response.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        return "无法识别图像内容"

# 替换Markdown文本中的图像连接为的图像识别与分析结果
def replace_image_with_text(markdown_text, image_dir_path):
    """
    将Markdown文本中的图像链接替换为图像识别与分析结果

    :param markdown_text: Markdown文本内容
    :param image_dir_path: 图像目录路径
    :return: 替换后的Markdown文本  
    """
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        image_text = image_to_text(full_path)
        return f"![{relative_path}]({image_text})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def replace_image_with_base64(markdown_text, image_dir_path):
    """
    将Markdown文本中的图像链接替换为base64编码的图像数据

    :param markdown_text: Markdown文本内容
    :param image_dir_path: 图像目录路径
    :return: 替换后的Markdown文本
    """
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages, is_ocr, layout_mode, formula_enable, table_enable, language):
    """
    将文件转换为Markdown格式，并进行相关处理和压缩

    :param file_path: 输入文件路径
    :param end_pages: 结束页码
    :param is_ocr: 是否进行OCR识别
    :param layout_mode: 布局模式
    :param formula_enable: 是否启用公式识别
    :param table_enable: 是否启用表格识别
    :param language: 语言
    :return: Markdown内容、纯文本内容、压缩包路径和新的PDF路径
    """
    try:
        file_path = to_pdf(file_path)
        # 获取识别的md文件以及压缩包文件路径
        local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1, is_ocr,
                                            layout_mode, formula_enable, table_enable, language)
        archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
        zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
        if zip_archive_success == 0:
            logger.info("压缩成功")
        else:
            logger.error("压缩失败")
        md_path = os.path.join(local_md_dir, file_name + ".md")
        with open(md_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        md_content = replace_image_with_text(txt_content, local_md_dir)
        # 返回转换后的PDF路径
        new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")
        return md_path, md_content, txt_content, archive_zip_path, new_pdf_path
    except Exception as e:
        logger.exception(e)


def init_model():
    """
    初始化模型（文本模型和OCR模型）

    :return: 0表示成功，-1表示失败
    """
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton

    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)
        logger.info(f"txt_model init final")
        ocr_model = model_manager.get_model(True, False)
        logger.info(f"ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def to_pdf(file_path):
    """
    将文件转换为PDF格式（如果不是PDF的话）

    :param file_path: 文件路径
    :return: PDF文件路径
    """
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            unique_filename = f"{uuid.uuid4()}.pdf"
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)
            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)
            return tmp_file_path


# 
def create_embeddings(texts, model="text-embedding-ada-002"):
    """
    Create embeddings for the given texts.
    
    Args:
        texts (str or List[str]): Input text(s)
        model (str): Embedding model name
        
    Returns:
        List[List[float]]: Embedding vectors
    """
    # Handle both string and list inputs
    input_texts = texts if isinstance(texts, list) else [texts]
    
    # Process in batches if needed (OpenAI API limits)
    batch_size = 100
    all_embeddings = []
    
    # Iterate over the input texts in batches
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]  # Get the current batch of texts
        
        # Create embeddings for the current batch
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        
        # Extract embeddings from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list
    
    # If input was a string, return just the first embedding
    if isinstance(texts, str):
        return all_embeddings[0]
    
    # Otherwise return all embeddings
    return all_embeddings

class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store text content
        self.metadata = []  # List to store metadata
    
    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.
        
        Args:
            text (str): The text content
            embedding (List[float]): The embedding vector
            metadata (Dict, optional): Additional metadata
        """
        self.vectors.append(np.array(embedding))  # Append the embedding vector
        self.texts.append(text)  # Append the text content
        self.metadata.append(metadata or {})  # Append the metadata (or empty dict if None)
    
    def add_items(self, items, embeddings):
        """
        Add multiple items to the vector store.
        
        Args:
            items (List[Dict]): List of text items
            embeddings (List[List[float]]): List of embedding vectors
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],  # Extract text from item
                embedding=embedding,  # Use corresponding embedding
                metadata={**item.get("metadata", {}), "index": i}  # Merge item metadata with index
            )
    
    def similarity_search_with_scores(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding with similarity scores.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return
            
        Returns:
            List[Tuple[Dict, float]]: Top k most similar items with scores
        """
        if not self.vectors:
            return []  # Return empty list if no vectors are stored
        
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]  # Compute cosine similarity
            similarities.append((i, similarity))  # Append index and similarity score
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results with scores
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # Retrieve text by index
                "metadata": self.metadata[idx],  # Retrieve metadata by index
                "similarity": float(score)  # Add similarity score
            })
        
        return results
    
    def get_all_documents(self):
        """
        Get all documents in the store.
        
        Returns:
            List[Dict]: All documents
        """
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]  # Combine texts and metadata

def create_bm25_index(chunks):
    """
    Create a BM25 index from the given chunks.
    
    Args:
        chunks (List[Dict]): List of text chunks
        
    Returns:
        BM25Okapi: A BM25 index
    """
    # Extract text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    
    # Tokenize each document by splitting on whitespace
    tokenized_docs = [text.split() for text in texts]
    
    # Create the BM25 index using the tokenized documents
    bm25 = BM25Okapi(tokenized_docs)
    
    # Print the number of documents in the BM25 index
    print(f"Created BM25 index with {len(texts)} documents")
    
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
            chunks.append(chunk_data)  # Add the chunk data to the list
    
    print(f"Created {len(chunks)} text chunks")  # Print the number of created chunks
    return chunks  # Return the list of chunks


def process_document(path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for fusion retrieval.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        Tuple[List[Dict], SimpleVectorStore, BM25Okapi]: Chunks, vector store, and BM25 index
    """
    if path.endswith('.pdf'):
        # Extract text from the PDF file
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(path)
    elif path.endswith('.md'):
        # Extract text from the MD file
        print("Extracting text from MD...")
        text = extract_text_from_md(path)
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

# 已失效
# def generate_response(query, context):
#     """
#     Generate a response based on the query and context.
    
#     Args:
#         query (str): User query
#         context (str): Context from retrieved documents
        
#     Returns:
#         str: Generated response
#     """
#     # Define the system prompt to guide the AI assistant
#     system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
#     If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

#     # Format the user prompt with the context and query
#     user_prompt = f"""Context:
#     {context}

#     Question: {query}

#     Please answer the question based on the provided context."""

#     # Generate the response using the OpenAI API
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-3.2-3B-Instruct",  # Specify the model to use
#         messages=[
#             {"role": "system", "content": system_prompt},  # System message to guide the assistant
#             {"role": "user", "content": user_prompt}  # User message with context and query
#         ],
#         temperature=0.1  # Set the temperature for response generation
#     )
    
#     # Return the generated response
#     return response.choices[0].message.content


latin_lang = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']

# 支持的语言列表（按类型分组）
all_lang = ['', 'auto']
all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])

latex_delimiters = [{"left": "$$", "right": "$$", "display": True},
                    {"left": '$', "right": '$', "display": False}]

Risk_item_data_query = [
    # 1. Financial risk
    "What is the financial risk status of this company? Please analyze its profitability (revenue growth, operating income) and debt ratio (debt divided by assets), and provide corresponding analysis.",
    # 2. Management&Operational risk
    "What is the management and operational risk status of this company? Please analyze the impact of supply chain disruptions and declining labor productivity on overall operations, as well as how governance issues might affect the company’s strategic direction.",
    # 3. Technological Obsolescence Risk
    "What is the status of technological obsolescence risk for this company? Please analyze how limited investment in R&D has increased its vulnerability in innovation-driven markets, and how the rapid pace of technological advancement and breakthrough R&D milestones constitute a core competitive advantage for the company.",
    # 4. Policy & Regulatory Risk
    "What are the policy and regulatory risks faced by this company? Please discuss how evolving legal requirements, stricter regulatory scrutiny, and increasing compliance costs affect its operational efficiency and financial performance, as well as the key role of tax benefits and policy-driven supply expansions in government initiatives that accelerate sectoral growth.",

    # 5. Liquidity risk
    "What is the liquidity risk status of this company? Please evaluate its short-term solvency and analyze the implications of the liquidity ratio declining to 0.50, particularly regarding the pressure on meeting near-term obligations.",
    
    # 6. Market & Macroeconomic Risk
    "What is the market and macroeconomic risk exposure for this company? Please analyze how the risks outlined in Item 1A are influenced by a range of macroeconomic variables, and how the company's economic sensitivity remains high, leading to significant revenue fluctuations in response to changing consumer spending and macroeconomic conditions.",

    # 7. Investment Liquidity Risk
    "What is the status of investment liquidity risk for this company? Please assess how the high concentration of non-current assets and long-term investments has increased the risk of asset immobilization and limited liquidity options.",

    # 8. Historical Performance Limitations
    "What limitations has the company's historical performance introduced regarding its ability to adapt to future market trends and demands? Please evaluate how this has affected its operating income, which reflects gains from core product sales, while non-operating income, such as gains on asset sales, is excluded from our recurring earnings metric due to its non-recurring nature."
    ]

Risk_item_Analysis_prompts = [

    "The debt-to-assets ratio assesses if the company's leverage is at an acceptable level.",
   
    "The management board identified supply chain disruptions and declining labor productivity effectively. Governance issues might Undermine the company’s strategic direction.",

    "The company’s limited investment in R&D has increased its vulnerability in innovation-driven markets. The rapid pace of technological advancement and breakthrough R&D milestones constitute a core competitive advantage for our company.",
    
    "The company is challenged by evolving legal requirements, stricter regulatory scrutiny, and increasing compliance costs (which may impact its operational efficiency and financial performance). Tax benefits and policy-driven supply expansions are key components of government initiatives accelerating sectoral growth.",
    
    "The company’s short-term solvency(the sufficient current assets to meet short-term liabilities) is under stress as the liquidity ratio declined to 0.50, reflecting heightened pressure on meeting near-term obligations.",

    "The risks outlined in Item 1A. are influenced by a range of macroeconomic variables. The company’s economic sensitivity remains high, as demand-supply elasticity indicates significant revenue fluctuations in response to changing consumer spending and macroeconomic conditions.",

    "The high concentration of non-current assets and long-term investments have increased the risk of asset immobilization and limited liquidity options.",
    
    "The company's past strategies have introduced limitations, restricting its ability to effectively adapt to future market trends and demands. Operating income reflects gains from core product sales, while Non-operating income, such as gains on asset sales, is excluded from our recurring earnings metric due to its non-recurring nature."
]


model_init = init_model()
logger.info(f"model_init: {model_init}")

with open("header.html", "r") as file:
    header = file.read()

with gr.Blocks() as demo:
    gr.HTML(header)
    
    # 新增状态变量
    vectorstore = gr.State()
    cleaned_texts = gr.State()
    
    with gr.Row():
        with gr.Column(variant='panel', scale=5):
            file = gr.File(label="Please upload a PDF or image", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
            max_pages = gr.Slider(1, 200, 20, step=1, label="Max convert pages")
            with gr.Row():
                layout_mode = gr.Dropdown([
                    # "layoutlmv3",
                    "doclayout_yolo"
                ], label="Layout model", value="doclayout_yolo")
                language = gr.Dropdown(all_lang, label="Language", value="auto")
            with gr.Row():
                formula_enable = gr.Checkbox(label="Enable formula recognition", value=True)
                is_ocr = gr.Checkbox(label="Force enable OCR", value=False)
                table_enable = gr.Checkbox(label="Enable table recognition(test)", value=True)
            with gr.Row():
                change_bu = gr.Button("Convert")
                clear_bu = gr.ClearButton(value="Clear")
            pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
            with gr.Accordion("Examples:"):
                example_root = os.path.join(os.path.dirname(__file__), "examples")
                gr.Examples(
                    examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                              _.endswith("pdf")],
                    inputs=file)


        with gr.Column(variant='panel', scale=5):
            output_file = gr.File(label="convert result", interactive=False)
            md_path = gr.Textbox(label="Markdown file path", interactive=False)
            with gr.Tabs():
                with gr.Tab("Markdown rendering"):
                    md = gr.Markdown(label="Markdown rendering（due to the GPU & bandwidth limit,it may take 1-2 minutes）", height=1100, show_copy_button=True,
                                     latex_delimiters=latex_delimiters, line_breaks=True)
                with gr.Tab("Markdown text"):
                    md_text = gr.TextArea(lines=45, show_copy_button=True)
            # with gr.Row():
            #     summary_output = gr.Textbox(label="Summary")
            #     generate_summary_button = gr.Button("Generate Summary")
            #     generate_summary_button.click(fn=generate_summary, inputs=[output_file], outputs=summary_output)
    file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
    change_bu.click(fn=to_markdown, inputs=[file, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
                    outputs=[md_path, md, md_text, output_file, pdf_show], api_name=False)
    clear_bu.add([file, md_path, md, pdf_show, md_text, output_file, is_ocr])

    # 先用slider模块设定好各种可变参数
    with gr.Accordion("RAG Search Settings"):
        k = gr.Slider(1, 20, 5, step=1, label="Top k:Number of results to return")
        alpha = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="Alpha:Vector/Keyword balance")
        chunk_size = gr.Slider(100, 5000, 1000, step=100, label="Chunk size")
        chunk_overlap = gr.Slider(0, 1000, 200, step=100, label="Chunk overlap")
    with gr.Row():
    # # 调用现有函数进行MD文档的处理和向量嵌入，点击按钮后会进行MD文档的处理和嵌入
        chunks, vector_store, bm25_index = gr.State(), gr.State(), gr.State()
    #     # md2vec_bu = gr.Button("Process Document")
    #     # clear_bu = gr.ClearButton(value="Clear")
    #     # md2vec_bu.click(fn=process_document, inputs=[md_path, chunk_size, chunk_overlap], outputs=[chunks, vector_store, bm25_index])
    #     # 嵌入完成后提示完成
    #     # clear_bu.add([chunks, vector_store, bm25_index])
        md_path.change(fn=process_document, inputs=[md_path, chunk_size, chunk_overlap], outputs=[chunks, vector_store, bm25_index])
        # 结束后在前端提示
        # 展示MD文档的处理和嵌入进度
        with gr.Column(variant='panel', scale=5):
            gr.Markdown("### Document Processing and Embedding Progress")
            gr.Markdown("This section displays the progress of document processing and embedding.")
            gr.Markdown("1. **Document Processing**: The document is processed and split into chunks.")
            gr.Markdown("2. **Embedding Creation**: The chunks are embedded using a model.")
            gr.Markdown("3. **Vector Store Creation**: The embeddings are stored in a vector store.")
            gr.Markdown("4. **BM25 Index Creation**: A BM25 index is created for keyword-based search.")
            gr.Markdown("5. **Ready for Search**: The document is ready for search queries.")
            gr.Markdown("6. **!!! This process may take 1 min~, please wait 1 min to do the righthand test.")
        # 调用现有函数进行RAG搜索测试
        with gr.Column(variant='panel', scale=5):
            gr.Markdown("### RAG Search Test")
            gr.Markdown("This section allows you to test the RAG search functionality.")
            gr.Markdown("1. **Enter Query**: Enter a query in the text box.")
            gr.Markdown("2. **Search**: Click the 'Search' button to perform the search.")
            gr.Markdown("3. **Results**: The top k results based on the search query will be displayed.")

            query = gr.Textbox(label="Enter your query here")
            search_bu = gr.Button("Search")
            search_results = gr.Dataframe(headers=["Score", "Chunk"], datatype=["number", "str"])

            # 定义一个辅助函数来处理 fusion_retrieval 的结果
            def process_search_results(query, chunks, vector_store, bm25_index, k, alpha):
                results = fusion_retrieval(query, chunks, vector_store, bm25_index, k, alpha)
                data = []
                for result in results:
                    chunk = result.get("text", "")
                    score = result.get("combined_score", 0.0)
                    data.append([score, chunk])
                return pd.DataFrame(data, columns=["Score", "Chunk"])

            search_bu.click(fn=process_search_results, inputs=[query, chunks, vector_store, bm25_index, k, alpha], outputs=[search_results])
            clear_bu.add([search_results])

            
    # 创建一个多选框组作为风险项查询提示词的多选择区
    with gr.Accordion("Prompt Selection"):
        gr.Markdown("### Prompt Selection")
        gr.Markdown("This section allows you to select prompts for search.")
        gr.Markdown("1. **Select Prompts**: Select the prompts you want to use for search.")
        gr.Markdown("2. **Search**: Click the 'Search' button to perform the search.")
        gr.Markdown("3. **Results**: The search results will be displayed.")
        prompt_checkboxes = []
        for template in Risk_item_data_query:
            checkbox = gr.Checkbox(label=template, value=False)
            prompt_checkboxes.append(checkbox)
        # 用于收集选中的提示模板
        selected_prompts = gr.State([])
        # 定义一个函数用于收集选中的提示模板
        def collect_selected_prompts(*checkbox_values):
            selected = []
            for i, value in enumerate(checkbox_values):
                if value:
                    selected.append(Risk_item_data_query[i])
            return selected
        
        # 当任意复选框状态改变时，更新选中的提示模板
        for checkbox in prompt_checkboxes:
            checkbox.change(
                fn=collect_selected_prompts,
                inputs=prompt_checkboxes,
                outputs=selected_prompts
            )
        # 当用户点击“Search”按钮时，循环生成selected_prompts的检索结果，并合并到search_results中
        search_button = gr.Button("Search")
        search_results = gr.Dataframe(headers=["Score", "Chunk"], datatype=["number", "str"])
        # 写一个循环处理selected_prompts的检索结果，并合并到search_results中的函数
        def generate_search_results(selected_prompts, chunks, vector_store, bm25_index, k, alpha):
            # 初始化一个空的DataFrame用于存储检索结果
            search_results = pd.DataFrame(columns=["Score", "Chunk"])
            # 循环处理selected_prompts的检索结果，并合并到search_results中
            for query in selected_prompts:
                # 调用fusion_retrieval函数进行检索
                results = fusion_retrieval(query, chunks, vector_store, bm25_index, k, alpha)
                # 提取所需的字段，调整顺序为 score 在前，chunk 在后
                chunk_score_pairs = [(result["combined_score"], result["text"]) for result in results]
                # 将检索结果转换为DataFrame
                results_df = pd.DataFrame(chunk_score_pairs, columns=["Score", "Chunk"])
                # 将检索结果与search_results合并
                search_results = pd.concat([search_results, results_df], ignore_index=True)
            # 返回合并后的检索结果
            return search_results
        # 当用户点击“Search”按钮时，生成相应的响应
        search_button.click(
            fn=generate_search_results,
            inputs=[selected_prompts, chunks, vector_store, bm25_index, k, alpha],  # 传入所需参数
            outputs=search_results 
        )
    # 根据搜索到的风险项相关数据，循环采用风险项分析提示词进行风险分析
    with gr.Accordion("Risk Analysis"):
        gr.Markdown("### Risk Analysis")
        gr.Markdown("This section allows you to analyze the risks based on the search results.")
        gr.Markdown("1. **Select Risk Items**: Select the risk items you want to analyze.")
        gr.Markdown("2. **Generate Analysis**: Click the 'Generate Analysis' button to perform the analysis.")
        gr.Markdown("3. **Analysis Results**: The analysis results will be displayed.")
        # 创建一个多选框组作为风险项查询提示词的多选择区
        risk_checkboxes = []
        for template in Risk_item_Analysis_prompts:
            checkbox = gr.Checkbox(label=template, value=False)
            risk_checkboxes.append(checkbox)
        # 用于收集选中的风险项
        selected_risk_items = gr.State([])
        # 定义一个函数用于收集选中的风险项
        def collect_selected_risk_items(*checkbox_values):
            selected = []
            for i, value in enumerate(checkbox_values):
                if value:
                    selected.append(Risk_item_data_query[i])
            return selected
        # 当任意复选框状态改变时，更新选中的风险项
        for checkbox in risk_checkboxes:
            checkbox.change(
                fn=collect_selected_risk_items,
                inputs=risk_checkboxes,
                outputs=selected_risk_items
            )
        # 当用户点击“Generate Analysis”按钮时，调用RAG循环生成相应的风险分析结果，合并后展示
        generate_analysis_button = gr.Button("Generate Analysis")
        results_tab = gr.Dataframe(headers=["Risk Item", "Analysis Result"], datatype=["str", "str"]) 
        # 定义一个函数用于生成风险分析结果
        risk_items = ["Financial risk","Management&Operational risk","Technological Obsolescence Risk","Policy & Regulatory Risk","Liquidity risk","Market and Macroeconomic Risk","Investment Liquidity Risk","Historical Performance Limitations"]
        
        # !!注意坑：gr.state类型的东西 要以click的函数调用方式才可以以list形式进行迭代，也就是这个东西要写到函数参数里用click调用传入，而不能直接放在函数里迭代，click应该是有一个state类处理器
        def generate_analysis_results(selected_risk_items, selected_prompts, chunks, vector_store, bm25_index, k, alpha):
            # 循环处理selected_risk_items的风险分析结果，并合并到analysis_results中
            analysis_result_list = []
            # 注意：selected_prompts, selected_risk_items 和 risk_item 长度相同，同时遍历这两个列表
            for prompt, query, risk_item in zip(selected_prompts, selected_risk_items, risk_items):
                # 调用fusion_retrieval函数进行检索
                ### todo：后面还可以优化，直接提取上面的检索结果，然后再进行分析，这样可以减少检索次数，提高效率
                results = fusion_retrieval(prompt, chunks, vector_store, bm25_index, k, alpha)
                # 提取检索结果中的文本
                chunk_texts = [result["text"] for result in results]
                # 将检索结果转换为字符串
                chunk_text = "\n".join(chunk_texts)
                ### todo:多个检索结果的合并，这里需要优化，因为每次检索都需要重新生成检索结果之间并被关联起来
                ### 关联起来的检索结果用于生成问题答案 可能效果会好很多
                
                # 写一个调用大模型进行风险分析的函数，这里需要传入检索结果和风险项
                def generate_analysis(query, chunk_text):
                    # 定义大模型的API调用
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash-preview-04-17",  # 选择合适的大模型
                        messages=[
                            {"role": "system", "content": "You are an expert in financial report analysis for listed companies. You can conduct data analysis based on the provided financial report information and provide analysis results for specified risk items."},  # 系统提示
                            {"role": "user", "content": f"There is some useful informations for you {chunk_text}, Please analyze the following risk item: {risk_item}. Please provide a detailed analysis of the risk item."}  # 用户提示
                        ]   
                    )
                    # 提取大模型的回复
                    analysis_result = response.choices[0].message.content
                    return analysis_result

                # 调用大模型进行风险分析
                analysis_result = generate_analysis(query, chunk_text)
                # 将分析结果累计存入list
                analysis_result_list.append(analysis_result)
            # 将风险项risk_item作为第一列，将analysis_result_list作为第二列添加到DataFrame中
            results_tab = pd.DataFrame({"Risk Item": risk_items, "Analysis Result": analysis_result_list})
            # 返回合并后的风险分析结果
            print('-------res:',results_tab)
            return results_tab
        # 当用户点击“Generate Analysis”按钮时，生成相应的风险分析结果
        generate_analysis_button.click(
            fn=generate_analysis_results,
            inputs=[selected_risk_items,  selected_prompts, chunks, vector_store, bm25_index, k, alpha],  # 传入所需参数
            outputs=results_tab  # 输出风险分析结果
        )      

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

