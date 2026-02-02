from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings


def build_vector_store() -> None:
    """Ingest the TXT knowledge base and build/update the FAISS index."""

    kb_file = Path(settings.kb_path)
    if not kb_file.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_file}")

    raw_text = kb_file.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    vectordb = FAISS.from_documents(docs, embeddings)

    output_dir = Path(settings.vectordb_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(output_dir))

    print(f"Vector store built with {len(docs)} chunks at {output_dir}")


if __name__ == "__main__":
    build_vector_store()
