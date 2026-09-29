"""오픈소스 임베딩 (설계 2.3). sentence-transformers 모델을 LangChain Embeddings로 감싼다.
Qwen3-Embedding처럼 질의용 프롬프트("query")가 있는 모델은 질의에만 프롬프트를 붙인다."""
import os
from functools import lru_cache

import torch
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

load_dotenv()
# huggingface_hub은 HF_TOKEN을 읽는다. 예전 이름(HUGGINGFACEHUB_API_TOKEN)으로 넣어도 동작하게 맞춘다
if not os.getenv("HF_TOKEN") and os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, device=_device())
        self.query_kwargs = {"prompt_name": "query"} if (self.model.prompts or {}).get("query") else {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True, **self.query_kwargs).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()
