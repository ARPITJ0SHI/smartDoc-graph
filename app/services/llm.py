"""LLM service with provider abstraction layer.

Supports Google Gemini via langchain-google-genai.
All calls are wrapped with tenacity retry logic (3 attempts, exponential backoff).
"""

import abc
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

# Retry config: 3 attempts, 1s → 2s → 4s backoff
_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)


class BaseLLMService(abc.ABC):
    """Abstract base for LLM providers."""

    @abc.abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Synchronous generation."""
        ...

    @abc.abstractmethod
    async def agenerate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Async generation for non-blocking calls in FastAPI event loop."""
        ...


class GoogleLLMService(BaseLLMService):
    """Google Gemini provider via langchain-google-genai."""

    def __init__(self) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = settings.llm_model or "gemini-3.1-flash-lite-preview"
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        logger.info("Google LLM initialized: %s", model)

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return "".join(texts)
        return str(content)

    @_RETRY_DECORATOR
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return self._extract_text(response.content)

    @_RETRY_DECORATOR
    async def agenerate(self, prompt: str, system_message: Optional[str] = None) -> str:
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        response = await self.llm.ainvoke(messages)
        return self._extract_text(response.content)


# ---- Singleton Factory ----

_llm_instance: Optional[BaseLLMService] = None


def get_llm_service() -> BaseLLMService:
    """Factory: returns the configured LLM service (singleton)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GoogleLLMService()
    return _llm_instance

