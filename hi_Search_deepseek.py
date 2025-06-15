import os
import logging
import numpy as np
import yaml
import pypdf
import docx
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
GLM_API_KEY = config['glm']['api_key']
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
GLM_URL = config['glm']['base_url']


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-3"
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


graph_func = HiRAG(
    working_dir=config['hirag']['working_dir'],
    enable_llm_cache=config['hirag']['enable_llm_cache'],
    embedding_func=GLM_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
    embedding_batch_num=config['hirag']['embedding_batch_num'],
    embedding_func_max_async=config['hirag']['embedding_func_max_async'],
    enable_naive_rag=config['hirag']['enable_naive_rag'])

def _read_pdf_text(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\\n"
    return text

def _read_docx_text(file_path: str) -> str:
    """Extracts text from a .docx file."""
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\\n'.join(full_text)

def load_document(file_path: str) -> str:
    """
    Loads a document from a file, supporting .txt, .pdf, and .docx.
    """
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".pdf":
        return _read_pdf_text(file_path)
    elif extension.lower() == ".docx":
        return _read_docx_text(file_path)
    elif extension.lower() == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {extension}")


# comment this if the working directory has already been indexed
# --- How to use ---
# 1. Set the path to your document
file_to_process = "your_document.pdf"  # Change to your .txt, .pdf, or .docx file

# 2. Load the content and insert it into HiRAG
try:
    document_content = load_document(file_to_process)
    graph_func.insert(document_content)
    print(f"Successfully processed and indexed {file_to_process}")
except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")
    print("Please update the 'file_to_process' variable with the correct path to your document.")


print("\\nPerform hi search:")
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="hi")))
