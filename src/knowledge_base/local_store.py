from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.KB_EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    vector_dir = Path(
        settings.KB_VECTORSTORE_DIR
    ).expanduser()

    if not vector_dir.exists():
        raise FileNotFoundError(
            f"本地知识库向量目录不存在: {vector_dir}，"
            "请先执行索引脚本"
        )

    vectorstore = Chroma(
        collection_name=settings.KB_COLLECTION_NAME,
        persist_directory=str(vector_dir),
        embedding_function=get_embeddings(),
    )

    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            f"本地知识库为空: {vector_dir}，"
            "请先执行索引脚本"
        )

    return vectorstore


def get_local_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.KB_TOP_K
        }
    )