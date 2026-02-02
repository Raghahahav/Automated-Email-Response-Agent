from typing import Optional

from langchain_core.runnables import Runnable

from config import settings
from rag_chain import build_email_rag_chain


_chain: Optional[Runnable] = None


def get_email_rag_chain() -> Runnable:
    """
    Lazily build and cache the email RAG chain.

    This keeps backend initialization separate from any specific frontend.
    """
    global _chain
    if _chain is None:
        _chain = build_email_rag_chain()
    return _chain


def draft_email_reply(email_text: str) -> str:
    """
    High-level helper that takes the raw client email text and returns
    the drafted reply as a plain string.
    """
    if not email_text.strip():
        raise ValueError("Email text is empty.")

    chain = get_email_rag_chain()
    result = chain.invoke({"email": email_text})

    # ChatGroq (LangChain) returns a BaseMessage with .content
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content

    # Fallback: stringify anything unexpected
    return str(result)
