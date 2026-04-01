"""LangChain wrappers for Gemini embedding and chat models."""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from multimodal_rag.config import CHAT_MODEL, EMBEDDING_DIMENSIONALITY, EMBEDDING_MODEL


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIMENSIONALITY,
    )


def get_chat() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0,
    )
