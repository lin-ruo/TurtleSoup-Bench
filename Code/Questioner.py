import asyncio
import json
import os
import re
from typing import List, Dict, Counter
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import google.generativeai as genai
from tools import *

class Questioner:
    def __init__(self, args, clients: Dict[str, AsyncOpenAI], model_configs: Dict, language: str,
                 key_clue_records_ref: List[Dict], blacklist_ref: set):
        self.args = args
        self.clients = clients
        self.model_configs = model_configs
        self.questioner_model_name = args.questioner_model
        self.language = language

        self.key_clue_records = key_clue_records_ref
        self.blacklist = blacklist_ref

        self.type_state = {
            "current_type": "unknown_type" if self.language == "zh" else "default",
            "confidence": 0.5,
            "last_checked": -1,
            "last_key_clues_count": 0
        }
        self.smoothing_alpha = 0.7
        self.switch_threshold = 0.1
        self.last_internal_summary = {}


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    async def _api_call(self, model_key: str, messages: List[Dict], **kwargs) -> str:
        client = self.clients.get(model_key)
        if not client:
            raise ValueError(f"Client for model key '{model_key}' not found in Questioner.")
        
        config = self.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Configuration for model key '{model_key}' not found in Questioner.")

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
            print(f"API Error [{actual_model_name} in Questioner]: {str(e)}")
            return ""

    def _format_o1_messages(self, messages: List[Dict]) -> List[Dict]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system"]
        if system_parts:
            combined_content = "\n\n".join(system_parts) + "\n\n" + "\n\n".join(user_parts)
        else:
            combined_content = "\n\n".join(user_parts)
        return [{"role": "user", "content": combined_content}]

    def _should_check_type(self, q_num: int, key_clues_count: int) -> bool:
        if q_num == 0 or self.type_state["last_checked"] == -1:
            return True
        new_clues = key_clues_count - self.type_state["last_key_clues_count"]
        return new_clues >= 3 or (q_num - self.type_state["last_checked"] >= 5)

    def _majority_vote(self, types: List[str]) -> tuple[str, float]:
        if not types:
            return ("unknown_type" if self.language == "zh" else "default"), 0.0
        if len(types) != 3:
            print(f"Warning: Majority vote called with {len(types)} types instead of 3.")
            if not types: return self.type_state["current_type"], self.type_state["confidence"]

        type_counts = Counter(types)
        if not type_counts:
            return self.type_state["current_type"], self.type_state["confidence"]

        most_common = type_counts.most_common(1)[0]
        majority_type, count = most_common
        
        vote_confidence = count / len(types) if types else 0.0

        if count == 1 and len(types) == 3:
            return self.type_state["current_type"], 0.33
        return majority_type, vote_confidence

    async def _determine_scenario_type(self, setup: str, history: List[Dict], current_key_clues: List[Dict]) -> str:
        history_str = format_history(history)
        clues_str = format_clue_records([f"Q: {c['question']} A: {c['answer']}" for c in current_key_clues]) # Format

        TEMPLATES = PROMPT_TEMPLATES_EN
        available_types = '/'.join(TEMPLATES.keys())

        prompt_filename = f"prompt_type_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)

        prompt = prompt_template.replace("{setup}", setup) \
                                .replace("{history_str}", history_str) \
                                .replace("{clues_str}", clues_str) \
                                .replace("{available_types}", available_types)

        messages = [{"role": "user", "content": prompt}]
        type_result = await self._api_call(self.questioner_model_name, messages, seed=42)
        cleaned_result = type_result.strip()

        for k in TEMPLATES.keys():
            if k in cleaned_result:
                return k
        return "default"

    async def _analyze_answer(self, last_answer_pair_str: str, history_for_analysis: List[Dict], setup: str) -> str:
        history_str = format_history(history_for_analysis)
        analyze_prompt_filename = f"prompt_analyze_{self.language}.txt"
        analyze_prompt_path = os.path.join("prompts", analyze_prompt_filename)
        analyze_prompt_template = load_prompt(analyze_prompt_path)

        analyze_prompt = analyze_prompt_template.replace("{answer}", last_answer_pair_str) \
                                                .replace("{history_str}", history_str) \
                                                .replace("{setup}", setup)
        return await self._api_call(self.questioner_model_name, messages=[{"role": "user", "content": analyze_prompt}], seed=42)

    async def _generate_internal_summary(self, setup: str, history: List[Dict], last_internal_summary_dict: Dict) -> Dict:
        clues = [f"Question: {rec['question']}\nAnswer: {rec['answer']}" for rec in self.key_clue_records if "<Key Clue>" in rec["answer"]]

        prompt_filename = f"prompt_summarize_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)

        last_summary_str = json.dumps(last_internal_summary_dict, ensure_ascii=False) if last_internal_summary_dict else ""

        prompt = prompt_template.replace("{surface}", setup) \
                                .replace("{tips}", format_clue_records(clues)) \
                                .replace("{question_log}", format_history(history)) \
                                .replace("{last_summary}", last_summary_str)

        summary_text = await self._api_call(
            model_key=self.questioner_model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=42
        )
        json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Internal summary JSON parsing error (Questioner): {e}, Original response: {summary_text}")
                return {"details": "", "logic": "", "conclusion": ""}
        else:
            print("Internal summary did not find JSON data (Questioner), falling back to default values")
            return {"details": "", "logic": "", "conclusion": ""}

    async def _propose_advice(self, setup: str, internal_summary: Dict) -> str:
        clues = [f"Question: {rec['question']}\nAnswer: {rec['answer']}" for rec in self.key_clue_records if "<Key Clue>" in rec["answer"]]

        prompt_filename = f"prompt_advice_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)
        advice_prompt = prompt_template.replace("{setup}", setup) \
                                       .replace("{clues}", format_clue_records(clues)) \
                                       .replace("{logic}", ''.join(internal_summary.get('logic', []))) \
                                       .replace("{details}", ''.join(internal_summary.get('details', []))) \
                                       .replace("{conclusion}", internal_summary.get('conclusion', ''))
                                       
        unexplained = await self._api_call(self.questioner_model_name, messages=[{"role": "user", "content": advice_prompt}], seed=42)
        return unexplained

    async def generate_question(self, setup: str, history: List[Dict], q_num: int) -> str:
        # ----------------------------------Learning and Refinement Module--------------------------------#
        # Learning from the new answer
        analysis = ""
        if history and len(history) >= 2: # ensure history has at least 2 elements (Q, A)
            last_q = history[-2]['content']
            last_a = history[-1]['content']
            last_answer_pair_str = f"Question: {last_q}\nAnswer: {last_a}"
            analysis = await self._analyze_answer(last_answer_pair_str, history[:-2], setup)
        
        advice = ""
        # Periodic summary and advice
        if q_num % self.args.summary_interval == 0 and q_num > 0:
            internal_summary_dict = await self._generate_internal_summary(setup, history, self.last_internal_summary)
            self.last_internal_summary = internal_summary_dict
            advice = await self._propose_advice(setup, internal_summary_dict)
        
        # # Ablation
        # analysis = ""
        # if history and len(history) >=2 :
        #     last_q = history[-2]['content']
        #     last_a = history[-1]['content']
        #     last_answer_pair_str = f"Question: {last_q}\nAnswer: {last_a}"
        #     analysis = ""
        # 
        # advice = ""
        # # Periodic summary and advice
        # if q_num % self.args.summary_interval == 0 and q_num > 0:
        #     internal_summary_dict = ""
        #     self.last_internal_summary = internal_summary_dict
        #     advice = ""
        
        #----------------------------------Learning and Refinement Module--------------------------------#

        history_questions = [h["content"] for h in history if h["type"] == "question"]
        history_questions_str = "\n".join(history_questions) if history_questions else ""


        #----------------------------------Type Experience Module--------------------------------#
        # Type verification logic
        current_key_clues_count = len(self.key_clue_records)
        if self._should_check_type(q_num, current_key_clues_count):
            types_determined = []
            if q_num == 0:
                self.type_state["current_type"] = "unknown_type" if self.language == "zh" else "default"
                self.type_state["confidence"] = 0.5
            else:
                for _ in range(3):
                    type_result = await self._determine_scenario_type(setup, history, self.key_clue_records)
                    types_determined.append(type_result)
                
                new_type, vote_confidence = self._majority_vote(types_determined)
                # Smooth the confidence
                smoothed_confidence = self.smoothing_alpha * self.type_state["confidence"] + (1 - self.smoothing_alpha) * vote_confidence

                if smoothed_confidence > self.type_state["confidence"] + self.switch_threshold:
                    self.type_state["current_type"] = new_type
                    self.type_state["confidence"] = smoothed_confidence
                else: # When not switching, fine-tune the confidence
                    if vote_confidence < self.type_state["confidence"]: # If the new vote result has lower confidence
                        self.type_state["confidence"] = max(0.3, smoothed_confidence) # Decrease it, but with a lower bound
                    else:
                        self.type_state["confidence"] = smoothed_confidence # Otherwise, smooth normally

            self.type_state["last_checked"] = q_num
            self.type_state["last_key_clues_count"] = current_key_clues_count
        
        # Get the prompt template path corresponding to the type
        current_type_key = self.type_state["current_type"]
        type_prompt_path_key = PROMPT_TEMPLATES_EN.get(current_type_key, {}).get("path", "default_prompt")
        
        type_prompt_filename = f"{type_prompt_path_key}_{self.language}.txt"
        type_prompt_full_path = os.path.join("prompts", "templates", type_prompt_filename)
        type_instruction_content = load_prompt(type_prompt_full_path)

        # # Ablation
        # type_instruction_content = ""
        #----------------------------------Type Experience Module--------------------------------#


        #----------------------------------Question Filtering Module--------------------------------#
        # Generate candidate questions
        question_gen_prompt_filename = f"prompt_question_{self.language}.txt"
        question_gen_prompt_path = os.path.join("prompts", question_gen_prompt_filename)
        question_gen_template = load_prompt(question_gen_prompt_path)

        formatted_history_for_prompt = format_history(history[-self.args.history_window_prompt:])

        prompt_for_questions = question_gen_template.replace("{surface}", setup) \
                                       .replace("{advice}", advice if advice else "") \
                                       .replace("{answer_analysis}", analysis if analysis else "") \
                                       .replace("{type_instruction}", type_instruction_content) \
                                       .replace("{history}", formatted_history_for_prompt)

        response_questions = await self._api_call(self.questioner_model_name, messages=[{"role": "user", "content": prompt_for_questions}], seed=42)
        
        candidate_questions = []
        for line in response_questions.split('\n'):
            line = line.strip()
            match = re.match(r"^\d+\.\s*(.*)", line)
            if match:
                question_text = match.group(1).strip()
                if "?" in question_text:
                    candidate_questions.append(question_text)

        # If there are fewer than 3 candidate questions, supplement them
        if not candidate_questions: # If no questions were extracted at all
            default_q_text = "Is the story related to a person?"
            candidate_questions.append(default_q_text)

        while len(candidate_questions) < 3:
            candidate_questions.append(candidate_questions[0])

        # Filter for the best question
        select_q_prompt_filename = f"prompt_select_question_{self.language}.txt"
        select_q_prompt_path = os.path.join("prompts", select_q_prompt_filename)
        select_q_template = load_prompt(select_q_prompt_path)

        blacklist_str = "- " + "\n- ".join(self.blacklist) if self.blacklist else ""
        formatted_history_for_selection = format_history(history[-self.args.history_window_selection:])

        prompt_for_selection = select_q_template.replace("{surface}", setup) \
                                                .replace("{advice}", advice if advice else "") \
                                                .replace("{answer_analysis}", analysis if analysis else "") \
                                                .replace("{history}", formatted_history_for_selection) \
                                                .replace("{history_questions}", history_questions_str) \
                                                .replace("{question1}", candidate_questions[0]) \
                                                .replace("{question2}", candidate_questions[1]) \
                                                .replace("{question3}", candidate_questions[2]) \
                                                .replace("{blacklist}", blacklist_str)

        best_question_response = await self._api_call(self.questioner_model_name, messages=[{"role": "user", "content": prompt_for_selection}], seed=42)
        
        # Parse the best question
        best_question = best_question_response.strip()
        
        # # Ablation
        # best_question = candidate_questions[0]

        # # Ensure the returned question is one of the candidates
        # if best_question not in candidate_questions:
        #     best_question = candidate_questions[0]  # Default to returning the first question
            
        #----------------------------------Question Filtering Module--------------------------------#
        return best_question

    async def generate_final_summary(self, setup: str, history: List[Dict], last_engine_summary: Dict = None) -> Dict:
        """
        Generates the final summary for evaluation.
        This method is called by the Engine at the end of the game.
        """
        clues = [f"Question: {rec['question']}\nAnswer: {rec['answer']}" for rec in self.key_clue_records if "<Key Clue>" in rec["answer"]]

        prompt_filename = f"prompt_summarize_{self.language}.txt"
        prompt_path = os.path.join("prompts", prompt_filename)
        prompt_template = load_prompt(prompt_path)

        last_summary_str = json.dumps(last_engine_summary, ensure_ascii=False) if last_engine_summary else ""

        prompt = prompt_template.replace("{surface}", setup) \
                                .replace("{tips}", format_clue_records(clues)) \
                                .replace("{question_log}", format_history(history)) \
                                .replace("{last_summary}", last_summary_str)

        summary_text = await self._api_call(
            model_key=self.questioner_model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=42,
            response_format={"type": "json_object"}
        )

        final_summary = parse_llm_json_safely(summary_text)

        if not final_summary.get("conclusion"):
            print(f"Warning: Final summary parsing failed or content is empty. Original response: {summary_text[:500]}...")
        
        return final_summary
        
def parse_llm_json_safely(text: str) -> dict:
    """
    A more robust JSON parser (V2), specifically for handling non-standard JSON that LLMs might generate.
    It bypasses parsing failures caused by unescaped quotes etc., by extracting key-value contents separately and reassembling them manually.
    This version no longer uses json.loads for the array parts, parsing them manually instead.
    """
    
    def extract_string_list(text_block: str) -> list:
        """A helper function to manually extract a list from the string content of a JSON array."""
        if not text_block:
            return []
        
        # Split by comma. For this use case, a simple split is sufficient
        # as elements are unlikely to contain commas.
        elements = text_block.split(',')
        
        cleaned_elements = []
        for el in elements:
            # Remove leading/trailing whitespace and potential quotes
            cleaned_el = el.strip()
            if cleaned_el.startswith('"') and cleaned_el.endswith('"'):
                cleaned_el = cleaned_el[1:-1]
            
            # Replace escape characters that the model might have used incorrectly
            cleaned_el = cleaned_el.replace('\\"', '"').replace('\\n', '\n')
            
            if cleaned_el: # Ensure empty strings are not added
                cleaned_elements.append(cleaned_el)
        return cleaned_elements

    try:
        # 1. Extract the string content of 'conclusion' (logic unchanged)
        conclusion_match = re.search(r'"conclusion"\s*:\s*"(.*)"\s*\}', text, re.DOTALL)
        conclusion_str = ""
        if conclusion_match:
            conclusion_str = conclusion_match.group(1).replace('\\"', '"').replace('\\n', '\n')

        # 2. Extract the raw content of the 'details' array
        details_match = re.search(r'"details"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        details_list = []
        if details_match:
            details_content = details_match.group(1)
            details_list = extract_string_list(details_content)

        # 3. Extract the raw content of the 'logic' array
        logic_match = re.search(r'"logic"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        logic_list = []
        if logic_match:
            logic_content = logic_match.group(1)
            logic_list = extract_string_list(logic_content)

        # 4. Safely reassemble into a dictionary in Python
        return {
            "details": details_list,
            "logic": logic_list,
            "conclusion": conclusion_str
        }
    except Exception as e:
        return {"details": [], "logic": [], "conclusion": ""}