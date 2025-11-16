import asyncio
import json
import os
import re
from typing import List, Dict

from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from tools import *

class Responder:
    def __init__(self, args, clients: Dict[str, AsyncOpenAI], model_configs: Dict, language: str,
                 allowed_answers: Dict, blacklist_ref: set):
        self.args = args
        self.clients = clients
        self.model_configs = model_configs
        self.responder_model_name = args.responder_model
        self.language = language
        self.allowed_answers = allowed_answers
        self.blacklist = blacklist_ref

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _api_call(self, model_key: str, messages: List[Dict], **kwargs) -> str:
        client = self.clients.get(model_key)
        if not client:
            raise ValueError(f"Client for model key '{model_key}' not found in Responder.")

        config = self.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Configuration for model key '{model_key}' not found in Responder.")
        
        actual_model_name = config['model']

        if 'o1' in actual_model_name.lower():
            messages = self._format_o1_messages(messages)
            
        try:
            response = await client.chat.completions.create(
                model=actual_model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error [{actual_model_name} in Responder]: {str(e)}")
            return ""

    def _format_o1_messages(self, messages: List[Dict]) -> List[Dict]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
        if system_parts:
            combined_content = "\n\n".join(system_parts) + "\n\n" + "\n\n".join(user_parts)
        else:
            combined_content = "\n\n".join(user_parts)
        return [{"role": "user", "content": combined_content}]

    def _normalize_answer(self, text: str) -> str:
        # Normalize text to one of "yes", "no", "unknown"
        text = text.strip("., ").lower()
        for key, values in self.allowed_answers.items():
            if any(v in text for v in values):
                return key # key is already "yes", "no", or "unknown"

        # Fallback for the 'unknown' case if not caught by the loop
        if any(kw in text for kw in self.allowed_answers.get("unknown", [])):
            return "unknown"
        
        print(f"Warning: Answer '{text}' could not be normalized to yes/no/unknown. Defaulting to unknown.")
        return "unknown"


    async def generate_answer_and_clue(self, setup: str, solution: str, tips: List[str], question: str) -> str:
        answer_prompt_filename = f"prompt_answer_{self.language}.txt"
        answer_prompt_path = os.path.join("prompts", answer_prompt_filename)
        answer_template = load_prompt(answer_prompt_path)

        prompt_for_answer = answer_template.replace("{surface}", setup).replace("{bottom}", solution)
        messages_for_answer = [
            {"role": "system", "content": prompt_for_answer},
            {"role": "user", "content": question}
        ]
        raw_answer = await self._api_call(
            self.responder_model_name,
            messages_for_answer,
            seed=42
        )
        normalized_answer = self._normalize_answer(raw_answer)

        if normalized_answer == "unknown":
            self.blacklist.add(question)

        # For ablation study, this part is removed.
        key_clue_prompt_filename = f"prompt_is_key_{self.language}.txt"
        key_clue_prompt_path = os.path.join("prompts", key_clue_prompt_filename)
        key_clue_template = load_prompt(key_clue_prompt_path)

        tips_str = ", ".join(tips) if tips else ""
        prompt_for_key_clue = key_clue_template.replace("{surface}", setup) \
                                               .replace("{bottom}", solution) \
                                               .replace("{tips}", tips_str) \
                                               .replace("{question}", question)
        
        response_is_key = await self._api_call(
            self.responder_model_name,
            messages=[{"role": "user", "content": prompt_for_key_clue}],
            seed=42
        )

        # # Ablation
        # response_is_key = ""

        is_key_clue = response_is_key.strip().lower().startswith("yes")
        
        if is_key_clue:
            return f"{normalized_answer}<Key Clue>"
        else:
            return normalized_answer