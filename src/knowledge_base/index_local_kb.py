import argparse
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import settings
from knowledge_base.local_store import get_embeddings


SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".json",
    ".pdf",
    ".docx",
}


def load_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        try:
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(path)).load()
            else:
                loaded = TextLoader(
                    str(path),
                    encoding="utf-8",
                ).load()

            for document in loaded:
                document.metadata.update(
                    {
                        "source": str(path),
                        "title": path.name,
                    }
                )

            documents.extend(loaded)

        except Exception as exc:
            print(f"跳过文件 {path}: {exc}")

    return documents


def make_chunk_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []

    for index, document in enumerate(documents):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        raw_id = (
            f"{source}:{index}:"
            f"{document.page_content}"
        )

        ids.append(
            hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()
        )

    return ids


def build_index(rebuild: bool = False) -> None:
    source_dir = Path(
        settings.KB_SOURCE_DIR
    ).expanduser()

    vector_dir = Path(
        settings.KB_VECTORSTORE_DIR
    ).expanduser()

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"知识库源目录不存在: {source_dir}"
        )

    documents = load_documents(source_dir)

    if not documents:
        raise RuntimeError(
            f"知识库目录中没有可读取的文件: {source_dir}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise RuntimeError("文件切分后没有得到有效文本块")

    vector_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rebuild:
        old_store = Chroma(
            collection_name=settings.KB_COLLECTION_NAME,
            persist_directory=str(vector_dir),
            embedding_function=get_embeddings(),
        )
        old_store.delete_collection()

    vectorstore = Chroma(
        collection_name=settings.KB_COLLECTION_NAME,
        persist_directory=str(vector_dir),
        embedding_function=get_embeddings(),
    )

    vectorstore.add_documents(
        documents=chunks,
        ids=make_chunk_ids(chunks),
    )

    print(
        f"索引建立完成："
        f"{len(documents)} 个文档，"
        f"{len(chunks)} 个文本块"
    )
    print(f"向量库目录：{vector_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除旧向量集合后重新建立索引",
    )

    args = parser.parse_args()

    build_index(
        rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()