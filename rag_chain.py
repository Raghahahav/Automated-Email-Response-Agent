from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from config import settings


def _load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    index_path = Path(settings.vectordb_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {index_path}. Run ingest.py first."
        )

    vectordb = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectordb.as_retriever(search_kwargs={"k": settings.k})


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _make_llm():
    if not settings.is_config_valid:
        raise RuntimeError("GROQ_API_KEY is not set. Please configure your .env file.")

    return ChatGroq(
        model=settings.groq_model_name,
        temperature=0.0,
    )


def build_email_rag_chain(user_name: str, company_name: str):
    """Builds a RAG chain that drafts compliant email responses from the KB only."""
    retriever = _load_retriever()
    llm = _make_llm()

    fallback_message = (
        "Thank you for reaching out. Based on the available information, this specific "
        "detail is not currently covered. Please allow us to connect you with our "
        "support team for further assistance.\n\n"
        f"Best regards,\n{user_name}\nAI Email Assistant\n{company_name}"
    )

    system_prompt = (
        "You are an AI email assistant for an Indian stock broking firm.\n"
        "You must answer strictly and only using the INTERNAL KNOWLEDGE BASE provided.\n"
        "If the knowledge base does not clearly cover the user's question, you MUST reply "
        f"exactly with the following fallback message and nothing else:\n\n{fallback_message}\n\n"
        "Rules:\n"
        "- Do not make up policies, numbers, timelines, or conditions that are not stated.\n"
        "- Do not use external or general knowledge.\n"
        "- Maintain a professional, compliance-safe, non-promissory tone.\n"
        "- Do not provide investment advice or guarantee returns.\n"
        "- Always answer in the form of an email reply.\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Here is the internal knowledge base context:\n\n{context}\n\n"
                "Here is the client's email:\n\n{email}\n\n"
                "Draft a clear, concise reply as per the rules above.",
            ),
        ]
    )

    # Build a small graph: retrieve docs -> format -> pass into prompt+llm
    retriever_chain = RunnableLambda(lambda x: retriever.invoke(x["email"]))

    rag_chain = (
        RunnableParallel(
            context=retriever_chain | RunnableLambda(_format_docs),
            email=RunnableLambda(lambda x: x["email"]),
        )
        | prompt
        | llm
    )

    return rag_chain
