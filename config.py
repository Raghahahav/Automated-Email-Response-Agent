import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """Application-level configuration."""

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
    )
    kb_path: str = os.getenv(
        "KB_PATH",
        "Referral & Client Engagement Policy – Internal Knowledge Base (v1.0).txt",
    )
    vectordb_path: str = os.getenv("VECTORDB_PATH", "data/faiss_index")
    k: int = int(os.getenv("RETRIEVAL_K", "4"))

    @property
    def is_config_valid(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
