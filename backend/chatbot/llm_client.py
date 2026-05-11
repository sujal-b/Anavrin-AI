"""
LLM Client - Proper integration with Nvidia NIM / OpenRouter APIs
with retry logic, timeout handling, and fallback to template responses.
"""

import os
import logging
import asyncio
from typing import Optional

import httpx

logger = logging.getLogger("chatbot.llm")


class LLMClient:
    """Handles LLM API calls with retry and fallback."""

    def __init__(
        self,
        provider: str = "mock",
        api_key: str = "",
        model: str = "",
        max_retries: int = 2,
        timeout: float = 30.0,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self.max_retries = max_retries
        self.timeout = timeout

    def _default_model(self) -> str:
        if self.provider == "nvidia":
            return "meta/llama-3.1-8b-instruct"
        elif self.provider == "openrouter":
            return "meta-llama/llama-3.1-8b-instruct:free"
        return ""

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "ML Customer Support Chatbot"

        return headers

    def _build_payload(self, messages: list[dict], max_tokens: int = 512) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

    def _get_url(self) -> str:
        if self.provider == "nvidia":
            return "https://integrate.api.nvidia.com/v1/chat/completions"
        elif self.provider == "openrouter":
            return "https://openrouter.ai/api/v1/chat/completions"
        return ""

    async def generate(
        self,
        user_message: str,
        intent: str,
        context: str = "",
        history: list[dict] = None,
        system_prompt: str = "",
        fallback_response: str = "",
    ) -> str:
        """
        Generate response using LLM API.
        Falls back to template response on failure.
        """
        if self.provider == "mock" or not self.api_key:
            return fallback_response or self._default_fallback(intent)

        default_system = f"""You are a helpful e-commerce customer support agent.
The user's intent has been classified as: {intent}.
Relevant context from knowledge base: {context}
Provide a helpful, concise response. Be specific and actionable.
Do not hallucinate order numbers or personal information."""

        messages = [
            {"role": "system", "content": system_prompt or default_system},
        ]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        url = self._get_url()
        headers = self._build_headers()
        payload = self._build_payload(messages)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            return content

                    logger.warning(f"LLM returned empty response (attempt {attempt + 1})")

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.warning(f"LLM timeout (attempt {attempt + 1}): {e}")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning(f"LLM HTTP error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM error (attempt {attempt + 1}): {e}")

            # Exponential backoff
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        logger.error(f"LLM failed after {self.max_retries + 1} attempts: {last_error}")
        return fallback_response or self._default_fallback(intent)

    def _default_fallback(self, intent: str) -> str:
        """Default fallback responses based on intent."""
        fallbacks = {
            "cancel_order": "To cancel your order, please provide your order number. You can cancel within 24 hours of placing the order.",
            "track_refund": "Refunds are typically processed within 5-7 business days. You can check your refund status in your account settings.",
            "get_refund": "To request a refund, please provide your order number and reason for the refund. We'll process it within 3-5 business days.",
            "check_payment_methods": "We accept credit cards, debit cards, PayPal, and bank transfers. All transactions are secure.",
            "delivery_options": "We offer standard (5-7 days), express (2-3 days), and same-day delivery options.",
            "contact_human_agent": "I'll connect you with a human agent. Please hold for a moment.",
            "check_invoice": "You can view and download your invoices from the 'Order History' section of your account.",
            "change_shipping_address": "To change your shipping address, please provide your order number and the new address. Changes are only possible before the order ships.",
        }
        return fallbacks.get(intent, "I'm here to help! Could you provide more details about your question?")


# Singleton instance
llm_client = LLMClient(
    provider=os.getenv("LLM_PROVIDER", "mock"),
    api_key=os.getenv("LLM_API_KEY", ""),
    model=os.getenv("LLM_MODEL", ""),
)
