"""오픈소스 임베딩 (설계 2.3). sentence-transformers 모델을 LangChain Embeddings로 감싼다.
Qwen3-Embedding처럼 질의용 프롬프트("query")가 있는 모델은 질의에만 프롬프트를 붙인다."""
import os
import threading
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

# 병렬 분석 노드(기술 요약·시장성)가 같은 모델을 동시에 호출하면 MPS에서 Segmentation fault가 난다.
# 모델 생성과 인코딩을 한 번에 하나씩만 하도록 잠근다 (검색 1회는 짧아서 대기 시간은 거의 없다)
_LOCK = threading.RLock()


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
        with _LOCK:
            return self.model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        with _LOCK:
            return self.model.encode(text, normalize_embeddings=True, **self.query_kwargs).tolist()


def get_embeddings() -> LocalEmbeddings:
    with _LOCK:
        return _load_embeddings()


@lru_cache(maxsize=1)
def _load_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()
