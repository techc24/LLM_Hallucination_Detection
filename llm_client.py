import json
import os
from typing import Any, Dict, List

import httpx


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class LLMClient:
    """
    Thin wrapper around OpenAI's Chat Completions API with a simple mock mode.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._api_key = os.getenv("OPENAI_API_KEY")

    def _headers(self) -> Dict[str, str]:
        if not self._api_key and not self.use_mock:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = httpx.post(OPENAI_API_URL, headers=self._headers(), json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    # --- Public methods ---

    def generate_answer(self, domain: str, question: str) -> str:
        if self.use_mock:
            return f"[MOCK ANSWER] In the domain of {domain}, a possible answer to: {question}"

        system_prompt = (
            "You are an expert assistant in high-stakes domains (finance, law, healthcare). "
            "Provide concise, factual answers. Avoid speculation."
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Domain: {domain}\nQuestion: {question}",
                },
            ],
            "temperature": 0.3,
        }

        data = self._post(payload)
        return data["choices"][0]["message"]["content"].strip()

    def decompose_claims(self, answer: str) -> List[str]:
        """
        Ask the LLM to decompose an answer into atomic factual claims.
        Returns a list of claim strings.
        """
        if self.use_mock:
            return [c.strip() for c in answer.split(".") if c.strip()]

        system_prompt = (
            "You extract atomic factual claims from answers.\n"
            "Each claim should be short, self-contained, and verifiable.\n"
            "Return ONLY a JSON array of strings, no explanations."
        )
        user_prompt = (
            "Decompose the following answer into atomic factual claims.\n\n"
            f"Answer:\n{answer}\n\n"
            "Return a JSON array of claim strings."
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        data = self._post(payload)
        content = data["choices"][0]["message"]["content"].strip()

        try:
            claims = json.loads(content)
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        except json.JSONDecodeError:

            pass


        return [c.strip() for c in answer.split(".") if c.strip()]

