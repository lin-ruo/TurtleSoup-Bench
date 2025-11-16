import asyncio
import os
import re
import json
from typing import List, Dict, Tuple
import numpy as np

from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tools import *

class Evaluator:
    def __init__(self, args, clients: Dict[str, AsyncOpenAI], model_configs: Dict, language: str):
        """
        Initializes the evaluator.
        Args:
            args: Namespace object containing configuration (e.g., evaluator_model, language).
            clients: Dictionary containing initialized AsyncOpenAI clients.
            model_configs: Dictionary containing model configurations.
            language: The current language being used ("zh" or "en").
        """
        self.args = args
        self.clients = clients
        self.model_configs = model_configs
        self.evaluator_model_key = args.evaluator_model
        self.language = language

        # Set hard upper and lower limits for N and M
        self.MIN_N_LOGIC = 2
        self.MAX_N_LOGIC = 5
        self.MIN_M_DETAILS = 3
        self.MAX_M_DETAILS = 8

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _api_call(self, model_key: str, messages: List[Dict], **kwargs) -> str:
        client = self.clients.get(model_key)
        if not client:
            raise ValueError(f"Client for model key '{model_key}' not found in Evaluator's clients dictionary.")

        config = self.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Configuration for model key '{model_key}' not found in Evaluator's model_configs.")

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
            print(f"API Error [{actual_model_name} in Evaluator]: {str(e)}")
            return ""

    def _format_o1_messages(self, messages: List[Dict]) -> List[Dict]:
        """Adapts the message format for o1 series models"""
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
        
        # Ensure the system prompt (if it exists) is at the beginning
        if system_parts:
            combined_content = "\n\n".join(system_parts) + "\n\n" + "\n\n".join(user_parts)
        else:
            combined_content = "\n\n".join(user_parts)
        return [{"role": "user", "content": combined_content}]

    def _calculate_heuristic_nm(self, text_length: int, num_sentences: int) -> tuple[int, int]:
        # --- Rules for Core Logic Points (N) ---
        if text_length < 180:
            n_logic = self.MIN_N_LOGIC
        elif text_length < 250:
            n_logic = 2
        elif text_length < 350:
            n_logic = 3
        elif text_length < 500:
            n_logic = 4
        else:
            n_logic = self.MAX_N_LOGIC

        # --- Rules for Core Detail Points (M) ---
        if text_length < 180:
            m_details = self.MIN_M_DETAILS
            if text_length >= 140:
                m_details = 4
        elif text_length < 250:
            m_details = 4
            if num_sentences >= 5 and text_length >= 200:
                m_details = 5
        elif text_length < 350:
            m_details = 5
            if num_sentences >= 7:
                m_details = 6
        elif text_length < 500:
            m_details = 6
            if num_sentences >= 8:
                m_details = 7
        else:
            m_details = self.MAX_M_DETAILS
        
        final_n = int(np.clip(n_logic, self.MIN_N_LOGIC, self.MAX_N_LOGIC))
        final_m = int(np.clip(m_details, self.MIN_M_DETAILS, self.MAX_M_DETAILS))
        
        return final_n, final_m

    async def _extract_key_points(self, true_solution_text: str) -> tuple[list, list]:
        """Extracts logical relationships and detailed information from the text"""
        text_length = len(true_solution_text)
        sentences = re.split(r'[.?!;\n]+', true_solution_text.strip())
        num_sentences = len([s for s in sentences if s.strip()])
        if num_sentences == 0 and text_length > 0: num_sentences = 1
            
        final_n, final_m = self._calculate_heuristic_nm(text_length, num_sentences)

        prompt_filename = f"prompt_extract_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)
        prompt = prompt_template.replace("{N_LOGIC_POINTS}", str(final_n)) \
                                .replace("{M_DETAIL_POINTS}", str(final_m)) \
                                .replace("{bottom}", true_solution_text)

        response = await self._api_call(
            model_key=self.evaluator_model_key,
            messages=[{"role": "user", "content": prompt}],
            seed=42
        )
        
        # Parse logic and details
        logic_section = []
        details_section = []
        current_section = None
        
        if not response:
            print("Warning: _extract_key_points received no response from API call.")
            return [], []

        for line in response.split("\n"):
            line = line.strip()

            if "[Logical Relationships]" in line:
                current_section = "logic"
            elif "[Detailed Information]" in line:
                current_section = "details"
            elif line.startswith("- ") and current_section:
                content = line[2:].strip()
                if content:  # Ensure empty lines are not added
                    if current_section == "logic":
                        logic_section.append(content)
                    elif current_section == "details":
                        details_section.append(content)
        
        # Handle the case of empty input
        if not logic_section and not details_section:
            return [], []
        
        return logic_section, details_section

    async def evaluate(self, true_solution: str, pred_summary: Dict) -> Dict:
        """Executes the evaluation process for logic and details"""
        true_logic, true_details = await self._extract_key_points(true_solution)
        
        pred_logic = pred_summary.get("logic", [])
        pred_details = pred_summary.get("details", [])
        pred_conclusion_input = pred_summary.get("conclusion", "")

        if isinstance(pred_conclusion_input, str):
            pred_conclusion_list_for_eval = [pred_conclusion_input] if pred_conclusion_input else []
        elif isinstance(pred_conclusion_input, list):
            pred_conclusion_list_for_eval = pred_conclusion_input
        else:
            pred_conclusion_list_for_eval = []


        logic_matches, unmatched_true_logic = await self._match_elements(true_logic, pred_logic)
        details_matches, unmatched_true_details = await self._match_elements(true_details, pred_details)

        conclusion_accuracy = 0.0
        # Only perform conclusion matching if both the true solution and predicted conclusion exist
        if true_solution.strip() and pred_conclusion_list_for_eval:
            conclusion_result = await self._get_semantic_similarity(true_solution, pred_conclusion_list_for_eval)
            conclusion_accuracy = conclusion_result['score'] if conclusion_result['matched'] else 0.0
        elif not pred_conclusion_list_for_eval:
            print("Warning: Predicted conclusion is empty, conclusion_accuracy set to 0.")

        logic_accuracy = sum(m["score"] for m in logic_matches) / len(true_logic) if true_logic else 0.0
        details_accuracy = sum(m["score"] for m in details_matches) / len(true_details) if true_details else 0.0

        logic_weight = 0.3
        details_weight = 0.3
        conclusion_weight = 0.4
        overall_score = (logic_accuracy * logic_weight) + \
                        (details_accuracy * details_weight) + \
                        (conclusion_accuracy * conclusion_weight)

        result = {
            "logic_accuracy": round(logic_accuracy, 2),
            "details_accuracy": round(details_accuracy, 2),
            "conclusion_match": round(conclusion_accuracy, 2),
            "overall_score": round(overall_score, 2),
            "details": {
                "ground_truth_logic": true_logic,
                "predicted_logic": pred_logic,
                "logic_matches": logic_matches,
                "unmatched_true_logic": unmatched_true_logic,
                "ground_truth_details": true_details,
                "predicted_details": pred_details,
                "details_matches": details_matches,
                "unmatched_true_details": unmatched_true_details,
                "predicted_conclusion_input": pred_conclusion_input,
                "true_solution_for_conclusion_eval": true_solution
            }
        }
        return result

    async def _match_elements(self, true_elements: List[str], pred_elements: List[str]) -> Tuple[List[Dict], List[str]]:
        if not true_elements:
            return [], []
        if not pred_elements:
            return [], list(true_elements)

        matches = []
        remaining_true_elements = list(true_elements)

        for t_elem in true_elements:
            result = await self._get_semantic_similarity(t_elem, pred_elements)

            if result["matched"]:
                final_score = 1.0 if result["score"] >= 0.8 else result["score"]
                matches.append({
                    "ground_truth": t_elem,
                    "predicted": result["predicted"],
                    "score": final_score
                })
                if t_elem in remaining_true_elements:
                    remaining_true_elements.remove(t_elem)

        return matches, remaining_true_elements

    async def _get_semantic_similarity(self, base_element: str, candidate_elements: List[str]) -> Dict:
        """
        Args:
            base_element (str): The base statement.
            candidate_elements (list): A list of candidate statements.

        Returns:
            dict: A dictionary containing the matching result.
                  On successful match: {"matched": True, "predicted": str, "score": float}
                  On no match:       {"matched": False, "score": float} (score is the actual highest score)
        """
        # Format the candidate list into a numbered string
        candidates_str = "\n".join([f'{idx+1}. "{cand}"' for idx, cand in enumerate(candidate_elements)])
        
        prompt_filename = f"prompt_similarity_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)
        prompt = prompt_template.replace("{ground_truth}", base_element) \
                                .replace("{predicted_list}", candidates_str)

        response_str = await self._api_call(
            model_key=self.evaluator_model_key,
            messages=[{"role": "user", "content": prompt}],
            seed=42,
            response_format={"type": "json_object"}
        )

        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not json_match:
            print(f"Warning: No valid JSON structure found in the similarity calculation API response. Original response: {response_str[:200]}")
        
        json_string = json_match.group(0)

        try:
            eval_result = json.loads(json_string)
            best_match_idx = eval_result.get("best_match_index")
            best_score = eval_result.get("best_match_score")

            if best_match_idx is not None and isinstance(best_match_idx, int) and \
               best_score is not None and isinstance(best_score, (int, float)) and \
               1 <= best_match_idx <= len(candidate_elements):
                
                # A valid best match was found
                matched_pred_element = candidate_elements[best_match_idx - 1]
                
                # If the score meets the relevance threshold, return the details of the successful match
                if best_score >= 0.5:
                    return {
                        "matched": True,
                        "predicted": matched_pred_element,
                        "score": round(best_score, 2)
                    }
            
            # If the index is null, or the score is below 0.5, it is considered not a match
            # We still return the highest score for future analysis
            return {"matched": False, "score": best_score if isinstance(best_score, (int, float)) else 0.0}

        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
            print(f"Warning: Failed to parse similarity calculation API response. Error: {e}.")
            return {"matched": False, "score": 0.0}