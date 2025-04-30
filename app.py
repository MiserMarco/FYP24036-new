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
api_key = 'sk-AKQSlxgw2L8tneJkdHbxfanF0hvkWiVlAdLBtLzUnjdqKnZX'
base_url = 'https://www.dmxapi.cn/v1'
client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
# 定义一个函数用于替换文本中的制表符为空格
def replace_t_with_space(texts):
    """
    将文本中的制表符替换为空格

    :param texts: 包含Document对象的列表，每个Document对象的page_content属性包含文本内容
    :return: 处理后的包含Document对象的列表
    """
    new_texts = []
    for text in texts:
        new_text = text.page_content.replace('\t', '')
        new_texts.append(Document(page_content=new_text, metadata=text.metadata))
    return new_texts


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
                        {"type": "text", "text": "请你对下面的图片进行内容识别与分析。"},
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


def encode_pdf_MD_and_get_split_documents(path, chunk_size=1000, chunk_overlap=200, progress=gr.Progress()):
    """
    对PDF或Markdown文件进行编码并获取分块后的文档
    :param path: 文件路径
    :param chunk_size: 分块大小
    :param chunk_overlap: 分块重叠大小
    :param progress: Gradio 进度条对象
    :return: 向量存储和清洗后的文本
    """
    start_time = time.time()
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path)
        documents = loader.load()
    elif path.endswith('.md'):
        # 加载md文档
        with open(path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        # 将txt_content转换为Document对象列表
        documents = [Document(page_content=txt_content, metadata={"source": path})]
    
    else:
        # print('???',path)
        raise ValueError(f"不支持的文件类型: {os.path.splitext(path)[1]}，目前仅支持 .pdf 和 .md 文件。")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    total_steps = len(texts)
    # 在进度更新处添加日志
    for i in progress.tqdm(range(total_steps), desc='Processing'):
        time.sleep(0.1)
        progress(i/total_steps, desc=f"Processing chunk {i+1}/{total_steps}")
    
    cleaned_texts = replace_t_with_space(texts)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key,
                                base_url=base_url)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    end_time = time.time()
    logger.info(f"向量化耗时: {end_time - start_time} 秒")
    return vectorstore, cleaned_texts


def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    创建BM25索引

    :param documents: 文档列表
    :return: BM25索引对象
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    结合关键词检索（BM25）和向量检索进行融合检索

    :param vectorstore: 向量存储
    :param bm25: BM25索引
    :param query: 查询字符串
    :param k: 返回文档数量
    :param alpha: 向量检索得分的权重
    :return: 排名前k的文档列表
    """
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    bm25_scores = bm25.get_scores(query.split())
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]


class FusionRetrievalRAG:
    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200, vectorstore = None, cleaned_texts = None, progress_callback=None):
        """
        初始化FusionRetrievalRAG类

        :param path: 文件路径
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠大小
        :param progress_callback: 进度回调函数
        """
        self.progress = progress_callback
        if vectorstore and cleaned_texts is not None:
            self.vectorstore, self.cleaned_texts = vectorstore, cleaned_texts
        else:
            self.vectorstore, self.cleaned_texts = encode_pdf_MD_and_get_split_documents(path, chunk_size, chunk_overlap,
                                                                                     progress_callback)
        self.bm25 = create_bm25_index(self.cleaned_texts)
        self.top_docs = []

    def run(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        执行融合检索

        :param query: 查询字符串
        :param k: 返回文档数量
        :param alpha: 向量检索得分的权重
        :return: 检索结果文档列表
        """
        self.top_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k, alpha)
        docs_content = [doc.page_content for doc in self.top_docs]
        show_context(docs_content)
        return self.top_docs


def parse_args():
    """
    解析命令行参数

    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Fusion Retrieval RAG Script")
    parser.add_argument('--path', type=str, default="../data/Understanding_Climate_Change.pdf",
                        help='Path to the PDF file.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of each chunk.')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Overlap between consecutive chunks.')
    parser.add_argument('--query', type=str, default='What are the impacts of climate change on the environment?',
                        help='Query to retrieve documents.')
    parser.add_argument('--k', type=int, default=5, help='Number of documents to retrieve.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for vector search vs. BM25.')

    return parser.parse_args()


def perform_rag_search(file_path, query, k=5, alpha=0.5, chunk_size=1000, chunk_overlap=200, vectorstore=None, cleaned_texts=None):
    if not file_path:
        return "Please upload a file first"

    try:
        rag = FusionRetrievalRAG(file_path,  chunk_size, chunk_overlap, vectorstore, cleaned_texts, gr.Progress())
        results = rag.run(query, k, alpha)
        return "\n\n------\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.exception(e)
        return f"Error during search: {str(e)}"

# 收集每一个Prompt的RAG结果

def merge_prompt_results(prompts, file_path, k, alpha, chunk_size, chunk_overlap, vectorstore = None, cleaned_texts = None, progress_callback=None):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key,base_url=base_url)
    # pdf_path = to_pdf(file_path.name)
    rag = FusionRetrievalRAG(file_path, chunk_size, chunk_overlap, vectorstore=None, cleaned_texts=None, progress_callback=gr.Progress())
    all_results = []
    for i, prompt in enumerate(gr.Progress().tqdm(prompts, desc='Processing Prompts')):
        results = rag.run(prompt, k, alpha)
        context = "\n\n".join([doc.page_content for doc in results])
        input_text = f"根据以下上下文内容：{context}，回答问题：{prompt}"
        output = llm.invoke(input_text).content
        all_results.append(f"Prompt {i + 1}: {prompt}\nAnswer: {output}\n")
    combined_results = "\n\n".join(all_results)
    analysis_prompt = f"请根据以下多个Prompt的回答结果进行综合分析：{combined_results}"
    final_analysis = llm.invoke(analysis_prompt).content
    return combined_results, final_analysis

# 获取处理后的Markdown文件前1000个字符
def get_markdown_summary(markdown_filename):
    encodings = ['utf-8', 'gbk', 'gb2312', 'big5']
    for encoding in encodings:
        try:
            with open(markdown_filename, 'r', encoding=encoding) as file:
                markdown_content = file.read()
                if len(markdown_content) > 2000:
                    return markdown_content[:2000]
                else:
                    return markdown_content
        except UnicodeDecodeError:
            continue
    return "无法使用支持的编码读取文件。"

# 使用大模型根据前1000个字符生成包含公司名称，年报年份的摘要，输出为json格式，包含公司名称，年报年份，其他信息
# 示例：{"company": "Company Name", "year": "2023", "other_info": "Additional information"}
def generate_summary(markdown_filename):
    llm = ChatOpenAI(model_name="gpt-4", api_key=api_key,base_url=base_url
                    )
    markdown_summary = get_markdown_summary(markdown_filename)
    prompt = f"根据以下内容生成包含公司名称，年报年份的摘要。包含公司名称，年报年份，和其他信息。\n\n{markdown_summary}"
    output = llm.invoke(prompt).content
    return output
    
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

prompt_templates = [
    # 1. 市场风险 (Market Risk)
    "Identify and analyze the market risks mentioned in the report, such as fluctuations in stock prices, interest rates, or commodity prices. How do these risks impact the company's revenue, profit margins, or investment portfolios? Provide specific examples from the financial statements or management discussion.",
    
    # 2. 信用风险 (Credit Risk)
    "Evaluate the company's credit risks, including potential defaults from customers, counterparties, or debtors. Focus on accounts receivable aging, debt repayment capabilities (e.g., current ratio, quick ratio), and any guarantees or collateral mentioned. Are there any significant credit-related contingencies disclosed?",
    
    # 3. 流动性风险 (Liquidity Risk)
    "Assess the company's liquidity risks by analyzing its ability to meet short-term obligations. Reference cash flow statements, working capital metrics (current assets vs. current liabilities), and debt maturity schedules. Does the report mention any reliance on short-term financing or potential challenges in accessing capital markets?",
    
    # 4. 操作风险 (Operational Risk)
    "Identify operational risks such as process inefficiencies, technology failures, or regulatory non-compliance. Focus on internal control disclosures, cybersecurity risks, supply chain disruptions, or legal/contractual issues. How does the company mitigate these risks, and what financial impacts are outlined in the report?",
    
    # 5. 汇率风险 (Foreign Exchange Risk)
    "Analyze foreign exchange risks faced by the company, especially if it has international operations or denominated liabilities/assets. Review currency hedging strategies, exposure to major currencies (e.g., USD, EUR, CNY), and the impact of exchange rate fluctuations on revenue, cost of goods sold, or foreign currency translation gains/losses.",
    
    # 6. 利率风险 (Interest Rate Risk)
    "Evaluate the company's exposure to interest rate fluctuations, particularly for debt instruments (loans, bonds) or interest-sensitive assets (e.g., cash equivalents, investments). How do changes in interest rates affect borrowing costs, interest income, or the fair value of financial instruments? Are there any interest rate hedging activities disclosed?",
    
    # 7. 合规风险 (Regulatory/Compliance Risk)
    "Assess regulatory risks such as changes in accounting standards (e.g., IFRS, GAAP), tax laws, or industry-specific regulations (e.g., SEC, GDPR for financial data). Identify any pending litigations, fines, or compliance failures mentioned in the report, and quantify their potential financial impacts (e.g., provision for penalties, restatements).",
    
    # 8. 战略风险 (Strategic Risk)
    "Analyze strategic risks related to business model sustainability, competitive positioning, or market disruption (e.g., new technologies, changing consumer preferences). Reference management’s discussion on growth strategies, R&D investments, or acquisitions/divestitures. How do these strategies expose the company to long-term financial risks, and what mitigation plans are outlined?",
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
                    md = gr.Markdown(label="Markdown rendering", height=1100, show_copy_button=True,
                                     latex_delimiters=latex_delimiters, line_breaks=True)
                with gr.Tab("Markdown text"):
                    md_text = gr.TextArea(lines=45, show_copy_button=True)
            with gr.Row():
                summary_output = gr.Textbox(label="Summary")
                generate_summary_button = gr.Button("Generate Summary")
                generate_summary_button.click(fn=generate_summary, inputs=[output_file], outputs=summary_output)



    file.change(fn=to_pdf, inputs=file, outputs=pdf_show)
    change_bu.click(fn=to_markdown, inputs=[file, max_pages, is_ocr, layout_mode, formula_enable, table_enable, language],
                    outputs=[md_path, md, md_text, output_file, pdf_show], api_name=False)
    clear_bu.add([file, md_path, md, pdf_show, md_text, output_file, is_ocr])

    # 新增RAG查询部分
    with gr.Accordion("RAG Search Settings"):
        
        rag_k = gr.Slider(1, 20, 5, step=1, label="Number of results")
        rag_alpha = gr.Slider(0, 1, 0.5, step=0.1, label="Vector/Keyword balance")
        chunk_size = gr.Slider(100, 1500, 1000, step=100, label="Chunk Size")
        chunk_overlap = gr.Slider(0, 200, 100, step=50, label="Chunk Overlap")
    
    # 调用现有函数执行MD文件的向量嵌入
    with gr.Row():
        embedding_progress = gr.Textbox(label="Embedding Progress", interactive=False)
        embedding_bu = gr.Button('Generate Embeddings')
        
        embedding_bu.click(
            fn=encode_pdf_MD_and_get_split_documents,
            inputs=[md_path, chunk_size, chunk_overlap],
            outputs=[vectorstore, cleaned_texts]
        )

    with gr.Accordion("RAG Search Test"):
        query = gr.Textbox(label="Enter your query", lines=2)
        search_btn = gr.Button("Search")
        search_results = gr.TextArea(label="Search Results", lines=10)
        search_btn.click(
            fn=perform_rag_search,
            inputs=[md_path, query, rag_k, rag_alpha, chunk_size, chunk_overlap],
            outputs=[search_results]
        )

    # 创建一个多选框组作为prompt多选择区
    with gr.Accordion("Prompt Selection"):
        prompt_checkboxes = []
        for template in prompt_templates:
            checkbox = gr.Checkbox(label=template, value=False)
            prompt_checkboxes.append(checkbox)
        
        # 用于收集选中的提示模板
        selected_prompts = gr.State([])
        
        # 定义一个函数用于收集选中的提示模板
        def collect_selected_prompts(*checkbox_values):
            selected = []
            for i, value in enumerate(checkbox_values):
                if value:
                    selected.append(prompt_templates[i])
            return selected
        
        # 当任意复选框状态改变时，更新选中的提示模板
        for checkbox in prompt_checkboxes:
            checkbox.change(
                fn=collect_selected_prompts,
                inputs=prompt_checkboxes,
                outputs=selected_prompts
            )
        
        # 生成结果的文本区域
        generated_results = gr.TextArea(label="Generated Results", lines=10)
        
        # 生成的结果的分析
        analysis_results = gr.TextArea(label="Analysis Results", lines=10)
        # 生成的结果的分析按钮
        generate_analysis_btn = gr.Button("Generate Analysis")
        # 生成的结果的分析按钮的点击事件
        generate_analysis_btn.click(
            fn=merge_prompt_results,
            inputs=[selected_prompts, md_path, rag_k, rag_alpha, chunk_size, chunk_overlap, vectorstore, cleaned_texts],
            outputs=[generated_results, analysis_results]
        )



            # # 生成的结果的分析
            # analysis_results = gr.TextArea(label="Analysis Results", lines=10)
            # # 生成的结果的分析按钮
            # generate_analysis_btn = gr.Button("Generate Analysis")
            # # 生成的结果的分析按钮的点击事件
            # generate_analysis_btn.click(
            #     fn=merge_prompt_results,
            #     inputs=[prompt_examples, md_path, rag_k, rag_alpha, chunk_size, chunk_overlap],
            #     outputs=[generated_results, analysis_results]
            # )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
    