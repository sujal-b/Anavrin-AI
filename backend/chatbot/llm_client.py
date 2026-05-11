"""
LLM Client - Proper integration with Nvidia NIM / OpenRouter APIs
with retry logic, timeout handling, and fallback to template responses.
"""

import os
import logging
import asyncio
from typing import Optional

import httpx

from backend.config.settings import config

logger = logging.getLogger("chatbot.llm")


class LLMClient:
    """Handles LLM API calls with retry and fallback."""

    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        max_retries: int = 2,
        timeout: float = 30.0,
    ):
        self.provider = provider or config.LLM_PROVIDER
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.max_retries = max_retries
        self.timeout = timeout
        
        logger.info(f"LLMClient initialized with provider: {self.provider}, model: {self.model}")

        # Override model if specific to provider but not set in config
        if not model and not config.LLM_MODEL:
            self.model = self._default_model()

    def _default_model(self) -> str:
        if self.provider.lower() == "nvidia":
            return "meta/llama-3.1-8b-instruct"
        elif self.provider.lower() == "openrouter":
            return "nvidia/nemotron-3-nano-30b-a3b:free"
        return ""

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.provider.lower() == "openrouter":
            headers["HTTP-Referer"] = "http://localhost:8000"
            headers["X-Title"] = "Anavrin AI Chatbot"

        return headers

    def _build_payload(self, messages: list[dict], max_tokens: int = 512) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.4,  # Slightly higher for more natural responses
        }

    def _get_url(self) -> str:
        provider_lower = self.provider.lower()
        if provider_lower == "nvidia":
            return "https://integrate.api.nvidia.com/v1/chat/completions"
        elif provider_lower == "openrouter":
            return "https://openrouter.ai/api/v1/chat/completions"
        return ""

    async def generate(
        self,
        user_message: str,
        intent: str,
        confidence: float = 1.0,
        context: str = "",
        history: list[dict] = None,
        system_prompt: str = "",
        fallback_response: str = "",
    ) -> str:
        """
        Generate response using LLM API.
        Falls back to template response on failure.
        """
        if self.provider.lower() == "mock" or not self.api_key:
            return fallback_response or self._default_fallback(intent)

        # Load production system prompt if not provided
        if not system_prompt:
            system_prompt = self._load_production_prompt(intent, context, confidence)

        # HARD GUARDRAIL: If confidence is very low and it's a generic intent, 
        # the ML model probably didn't find a match. 
        if confidence < 0.40 and intent in ["contact_customer_service", "general_inquiry"]:
             return "I apologize, but I am specialized only in e-commerce support (orders, refunds, shipping, and payments). I don't have information on that topic. How can I help you with your purchase today?"

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if history:
            # Only keep the last 10 messages for token efficiency
            messages.extend(history[-10:])

        messages.append({"role": "user", "content": user_message})

        url = self._get_url()
        if not url:
            return fallback_response or self._default_fallback(intent)

        headers = self._build_headers()
        payload = self._build_payload(messages)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    
                    if response.status_code == 401:
                        logger.error("LLM API Key is invalid or missing.")
                        break
                        
                    response.raise_for_status()
                    data = response.json()

                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            return content.strip()

                    logger.warning(f"LLM returned empty response (attempt {attempt + 1})")

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            except Exception as e:
                last_error = str(e)

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)

        logger.error(f"LLM failed: {last_error}")
        return fallback_response or self._default_fallback(intent)

    def _load_production_prompt(self, intent: str, context: str, confidence: float) -> str:
        """Loads, formats, and prioritizes the production-grade system prompt."""
        prompt_path = os.path.join(config.BASE_DIR, "ml_customer_support_system_prompt_v2.md")
        
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Strip metadata/ToC if they exist (lines starting with # or ## before CORE IDENTITY)
                # This keeps the prompt focused on instructions
                if "## CORE IDENTITY" in content:
                    content = content[content.find("## CORE IDENTITY"):]

                # Replace placeholders
                content = content.replace("[COMPANY_NAME]", "Anavrin AI")
                
                # Build a HIGH-PRIORITY runtime context block
                # We put this at the TOP so the LLM sees the current state immediately
                directive = f"""### STRICT GUARDRAIL: YOU ARE A SUPPORT BOT
YOU ARE AUTHORIZED ONLY TO ASSIST WITH E-COMMERCE (Orders, Refunds, Shipping, Payments).
DO NOT answer general knowledge, geography, or unrelated academic questions.

- CURRENT_INTENT: {intent}
- CLASSIFIER_CONFIDENCE: {confidence:.2f}
- KNOWLEDGE_CONTEXT: {context if context else 'No specific context found.'}

OPERATING RULES:
1. IF THE QUERY IS NOT ABOUT E-COMMERCE -> REFUSE POLITELY.
2. IF CLASSIFIER_CONFIDENCE < 0.50 -> ESCALATE PER CONSTRAINT 3.
3. DO NOT BE A GENERAL ASSISTANT. BE A SUPPORT AGENT FOR ANAVRIN AI.
--------------------------------------------------

"""
                return directive + content
        except Exception as e:
            logger.error(f"Failed to load production prompt: {e}")

        return f"You are Anavrin AI. Intent: {intent}. Context: {context}. Confidence: {confidence}"

    def _default_fallback(self, intent: str) -> str:
        """Default fallback responses based on intent."""
        fallbacks = {
            "cancel_order": "I can help you cancel your order. Please provide your order number. Note that cancellations are only possible before the order has been processed for shipping.",
            "get_refund": "To process a refund, I'll need your order number. Once submitted, refunds typically take 5-7 business days to appear in your account.",
            "track_order": "You can track your order using the tracking link sent to your email, or provide your order ID here so I can check the status for you.",
            "delivery_options": "We offer Standard (5-7 days), Express (2-3 days), and Next-Day delivery. Which one would you like to know more about?",
            "check_payment_methods": "We accept all major credit cards, PayPal, and Apple Pay for secure transactions.",
        }
        return fallbacks.get(intent, "I'm Anavrin AI, your support assistant. How can I help you with your e-commerce needs today?")


# Singleton instance initialized with config values
llm_client = LLMClient()
