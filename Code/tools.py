from configparser import ConfigParser
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
from pathlib import Path 

# 中文提示模板定义
PROMPT_TEMPLATES_ZH = {
    "罪案惊悚类": {
        "path": "Crime_Thriller_prompt",
        "description": "此类故事的结局聚焦于由人为恶意引发的犯罪行为，如谋杀、伤害、绑架、虐待等。其核心在于展现暴力、残酷或令人恐惧的情节，强调外部行为造成的直接冲击和悬疑恐怖感。"
    },
    "心智迷局类": {
        "path": "Mind_Game_prompt",
        "description": "此类故事的反转根源于角色异常的内部精神世界。结局往往由精神疾病（幻觉、妄想、人格分裂）、极端心理状态（偏执、痴迷）、扭曲的认知或动机所驱动。重点在于探索人物内在的混乱、困惑或非理性。"
    },
    "灵异奇幻类": {
        "path": "Supernatural_Fantasy_prompt",
        "description": "此类故事的解答涉及超脱于现实逻辑和科学解释的力量或现象。结局可能包含鬼魂、神魔、魔法、诅咒、预知、轮回等超自然元素，或是时间旅行、平行宇宙等科幻设定，核心在于非现实的、神秘的或怪诞的元素。"
    },
    "世事无常类": {
        "path": "Constant_Change_prompt",
        "description": "此类故事的结局并非简单源于纯粹的恶意或超自然，而是由不幸的意外、宿命的讽刺、深刻的误解、复杂的情感（爱、恨、内疚、牺牲）或特定的困境所造成。结局往往带有悲剧色彩，引人感伤、同情或对命运、人性的反思。"
    },
    "逻辑巧思类": {
        "path": "Clever_Logic_prompt",
        "description": "此类故事的重点在于解谜过程中的思维方式和结局的巧妙性。结局通常依赖于突破常规的逻辑推理、关键信息的视角转换（如叙述者身份、特定场景设定），或是揭示看似怪异现象背后符合常理的、甚至平凡而又感动的真相。有时带有黑色幽默或智力游戏的意味。"
    },
    "未知类型": { 
        "path": "default_prompt",
        "description": "类型未知，不在其余范畴内。"
    }
}

PROMPT_TEMPLATES_EN = {
    "Crime Thriller Type": {
        "path": "Crime_Thriller_prompt", 
        "description": "The ending of this type of story focuses on criminal acts caused by human malice, such as murder, injury, kidnapping, abuse, etc. Its core lies in presenting violent, cruel, or frightening plots, emphasizing the direct impact and suspenseful horror of external actions."
    },
    "Mind Maze Type": {
        "path": "Mind_Game_prompt",
        "description": "The twist in this type of story stems from the character's abnormal internal mental world. The ending is often driven by mental illness (hallucinations, delusions, multiple personalities), extreme psychological states (paranoia, obsession), or distorted cognitions or motives. The focus is on exploring the character's internal chaos, confusion, or irrationality."
    },
    "Supernatural Fantasy Type": {
        "path": "Supernatural_Fantasy_prompt",
        "description": "The solution to this type of story involves powers or phenomena that transcend realistic logic and scientific explanation. The ending may include supernatural elements like ghosts, gods and demons, magic, curses, precognition, reincarnation, or sci-fi settings like time travel or parallel universes. The core lies in unrealistic, mysterious, or grotesque elements."
    },
    "Worldly Vicissitudes Type": { 
        "path": "Constant_Change_prompt",
        "description": "The ending of this type of story does not simply stem from pure malice or the supernatural, but is caused by unfortunate accidents, fateful irony, profound misunderstandings, complex emotions (love, hate, guilt, sacrifice), or specific dilemmas. The ending often has a tragic color, evoking sadness, sympathy, or reflection on fate and humanity."
    },
    "Logic and Cleverness Type": {
        "path": "Clever_Logic_prompt",
        "description": "The focus of this type of story is on the way of thinking during the puzzle-solving process and the ingenuity of the ending. The ending usually relies on unconventional logical reasoning, perspective shifts of key information (such as narrator identity, specific scene settings), or revealing the common-sense, even ordinary and moving truths behind seemingly bizarre phenomena. Sometimes it carries a sense of black humor or intellectual games."
    },
    "default": { 
        "path": "default_prompt",
        "description": "Type unknown, not within other categories."
    }
}

ALLOWED_ANSWERS_ZH = {
    "yes": ["是", "对", "有"], 
    "no": ["否", "错", "没"],  
    "unknown": ["未知", "不知道", "无关"] 
}

ALLOWED_ANSWERS_EN = {
    "yes": ["yes", "correct", "true", "yep", "yeah"],
    "no": ["no", "false", "incorrect", "nope"],
    "unknown": ["unknown", "not sure", "irrelevant", "don't know", "dunno"]
}

def load_scenarios(json_path: str) -> List[Dict]:
    """
    Loads Turtle Soup scenario data from the specified JSON file.
    Each scenario should contain "title", "surface" (the setup), and "bottom" (the solution).
    "tips" are optional.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            scenarios_raw = json.load(f)
        
        # Convert and validate the scenario data structure
        processed_scenarios = []
        for i, s_data in enumerate(scenarios_raw):
            if not isinstance(s_data, dict):
                print(f"Warning: Scenario data at index {i+1} is not a dictionary, skipping. Path: {json_path}")
                continue
            
            title = s_data.get("title")
            surface = s_data.get("surface")
            bottom = s_data.get("bottom")

            if not all([title, surface, bottom]):
                print(f"Warning: Scenario '{title or f'Unnamed Scenario {i+1}'}' is missing title, surface, or bottom fields, skipping. Path: {json_path}")
                continue
                
            processed_scenarios.append({
                "title": str(title),
                "setup": str(surface),
                "solution": str(bottom),
                "tips": s_data.get("tips", [])
            })
        return processed_scenarios
    except FileNotFoundError:
        print(f"Error: Scenario file not found at path {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to parse scenario file {json_path}, please check the JSON format.")
        return []
    except Exception as e:
        print(f"An unknown error occurred while loading scenarios from {json_path}: {e}")
        return []


def load_prompt(prompt_filepath: str) -> str:
    """Loads prompt text from the specified file."""
    try:
        with open(prompt_filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at path {prompt_filepath}")
        return "" # Return an empty string or raise an exception, depending on your error handling strategy
    except Exception as e:
        print(f"An unknown error occurred while loading the prompt file {prompt_filepath}: {e}")
        return ""


def validate_answer(answer: str) -> bool:
    """
    Validates if the answer string conforms to the expected format (including whether it has a key clue tag).
    """
    plain_answers_en = {"yes", "no", "unknown"}
    clue_answers_en = {"yes<Key Clue>", "no<Key Clue>", "unknown<Key Clue>"}
    
    valid_answers = plain_answers_en.union(clue_answers_en)
    
    return answer in valid_answers


def format_clue_records(clues: List[str]) -> str:
    """
    Formats a list of key clue records for display or as input for a language model.
    clues: A list of strings, where each string represents a clue.
    """
    if not clues:
        return "None"
    # Use a more generic arrow or marker
    return "\n".join(f"  - {clue}" for clue in clues)


def format_history(history: List[Dict]) -> str:
    """
    Formats a list of conversation history for display or as input for a language model.
    history: A list of dictionaries, where each dictionary represents a turn in the conversation, expected to contain "type" and "content".
    """
    if not history:
        return "No conversation history"

    formatted_entries = []
    for i, msg in enumerate(history):
        turn_number = i // 2 + 1 # Every two messages (one question, one answer) count as one turn
        msg_type = msg.get("type", "unknown_type")
        content = msg.get("content", "")

        # Select a label based on the message type
        if msg_type == "question":
            label = f"Question {turn_number}"
        elif msg_type == "answer":
            label = f"Answer {turn_number}"
        elif msg_type == "conclusion":
            label = "Conclusion"
            # For conclusions, the content might be a dictionary and needs special handling
            if isinstance(content, dict):
                # Simply convert it to a JSON string, or format it selectively
                content_str = json.dumps(content, ensure_ascii=False, indent=2)
                # Or a more concise representation:
                # content_str = f"Logic: {content.get('logic')}, Details: {content.get('details')}, Conclusion: {content.get('conclusion')}"
            else:
                content_str = str(content)
            formatted_entries.append(f"{label}: {content_str}")
            continue
        else:
            label = f"{msg_type.capitalize()} {turn_number}"
        
        formatted_entries.append(f"{label}: {content}")
        
    return "\n".join(formatted_entries)


def load_config(config_filepath: str = 'config.ini') -> Dict:
    """
    Loads model configurations from the specified .ini file.
    Returns a dictionary where keys are section names and values are dictionaries containing 'model', 'base_url', and 'api_key'.
    """
    config = ConfigParser()
    if not os.path.exists(config_filepath):
        print(f"Error: Configuration file {config_filepath} not found.")
        # You can choose to raise an exception or return an empty dictionary
        # raise FileNotFoundError(f"Configuration file {config_filepath} not found.")
        return {}
        
    try:
        config.read(config_filepath, encoding='utf-8')
    except Exception as e:
        print(f"Error reading configuration file {config_filepath}: {e}")
        return {}

    model_configs = {}
    for section in config.sections():
        try:
            model_configs[section] = {
                'model': config.get(section, 'model'),
                'base_url': config.get(section, 'base_url'),
                'api_key': config.get(section, 'api_key')
            }
        except Exception as e:
            print(f"Section '{section}' in the config file is missing required keys (model, base_url, api_key) or there was a reading error: {e}")
            continue
            
    return model_configs



class RunLogger:
    def __init__(self, log_dir: str, run_id: str, model_pair_name: str):
        """
        Initializes the logger.
        Args:
            log_dir (str): The root directory where log files are stored.
            run_id (str): A unique identifier for this run.
                          It can also be a combination of the dataset name and the model combination.
            model_pair_name (str): The name of the current model combination being run (e.g., "Q-glm4_R-glm4").
        """
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.model_pair_name = model_pair_name
        
        # Create a separate log file for each model combination and dataset
        self.log_file_path = self.log_dir / f"run_log_{self.run_id}_{self.model_pair_name}.txt"
        self.summary_log_file_path = self.log_dir / f"summary_log_{self.run_id}_{self.model_pair_name}.txt" # For periodic summaries

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._write_log_header()

        self.processed_scenarios_count = 0
        self.cumulative_eval_metrics = {
            "logic_accuracy": 0.0,
            "details_accuracy": 0.0,
            "conclusion_match": 0.0,
            "overall_score": 0.0,
            "count": 0
        }
        self.log_summary_interval = 10 

    def _write_log_header(self):
        """Writes the log file header if the file is newly created."""
        if not self.log_file_path.exists():
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"===== Run Log: {self.run_id} | Models: {self.model_pair_name} | Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                f.write("Timestamp | Dataset | Scenario Title | Status | Questions | Duration(s) | Overall Score | Logic Acc | Details Acc | Conclusion Match | Error\n")
                f.write("-" * 150 + "\n")
        if not self.summary_log_file_path.exists():
             with open(self.summary_log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"===== Summary Log: {self.run_id} | Models: {self.model_pair_name} | Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                f.write("Timestamp | Processed Count | Avg Overall | Avg Logic | Avg Details | Avg Conclusion\n")
                f.write("-" * 100 + "\n")


    def log_initial_batch_info(self, datasets_info: Dict[str, List[str]]):
        """
        Logs information at the start of a batch process, including all planned datasets and scenarios.
        Args:
            datasets_info (Dict[str, List[str]]): A dictionary where keys are dataset filenames and values are lists of scenario titles planned to run in that dataset.
        """
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Initial Batch Information ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            for dataset_file, scenario_titles in datasets_info.items():
                f.write(f"Dataset: {dataset_file}\n")
                if scenario_titles:
                    for i, title in enumerate(scenario_titles):
                        f.write(f"  - Planned Scenario {i+1}: {title}\n")
                else:
                    f.write(f"  - No scenarios planned or loaded for this dataset.\n")
            f.write("-" * 150 + "\n\n")

    def log_scenario_result(self, dataset_name: str, result: Dict):
        """
        Logs the result of a single scenario's processing.
        Args:
            dataset_name (str): The name of the dataset to which the current scenario belongs.
            result (Dict): The result dictionary returned by TurtleSoupEngine.run_scenario.
        """
        meta = result.get("metadata", {})
        eval_data = result.get("evaluation", {})

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title = meta.get("title", "N/A")
        status = meta.get("status", "N/A")
        questions = meta.get("questions_used", 0)
        duration = meta.get("duration_sec", 0.0)
        error_msg = meta.get("error", "") if status == "error" else ""

        overall = eval_data.get("overall_score", 0.0) if status != "error" else 0.0
        logic = eval_data.get("logic_accuracy", 0.0) if status != "error" else 0.0
        details = eval_data.get("details_accuracy", 0.0) if status != "error" else 0.0
        conclusion = eval_data.get("conclusion_match", 0.0) if status != "error" else 0.0
        
        log_entry = (
            f"{timestamp} | {dataset_name} | {title} | {status.upper()} | "
            f"{questions} | {duration:.1f}s | "
            f"{overall:.2f} | {logic:.2f} | {details:.2f} | {conclusion:.2f} | "
            f"{error_msg}\n"
        )
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        # Update cumulative data for calculating periodic averages
        if status != "error" and isinstance(eval_data.get("overall_score"), (int, float)):
            self.cumulative_eval_metrics["logic_accuracy"] += logic
            self.cumulative_eval_metrics["details_accuracy"] += details
            self.cumulative_eval_metrics["conclusion_match"] += conclusion
            self.cumulative_eval_metrics["overall_score"] += overall
            self.cumulative_eval_metrics["count"] += 1
        
        self.processed_scenarios_count += 1

        # Check if it's time to log a periodic summary
        if self.processed_scenarios_count % self.log_summary_interval == 0 and self.cumulative_eval_metrics["count"] > 0:
            self.log_periodic_summary()

    def log_periodic_summary(self):
        """Logs the periodic average evaluation results."""
        count = self.cumulative_eval_metrics["count"]
        if count == 0:
            return # No data to summarize

        avg_overall = self.cumulative_eval_metrics["overall_score"] / count
        avg_logic = self.cumulative_eval_metrics["logic_accuracy"] / count
        avg_details = self.cumulative_eval_metrics["details_accuracy"] / count
        avg_conclusion = self.cumulative_eval_metrics["conclusion_match"] / count
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary_entry = (
            f"{timestamp} | {self.processed_scenarios_count} (total processed) | "
            f"{avg_overall:.4f} | {avg_logic:.4f} | {avg_details:.4f} | {avg_conclusion:.4f}\n"
        )
        with open(self.summary_log_file_path, 'a', encoding='utf-8') as f:
            f.write(summary_entry)
        

    def log_batch_completion(self, batch_summary_data: Dict):
        """
        Logs the final summary after an entire batch process is complete (e.g., one model pair's run on all datasets).
        Args:
            batch_summary_data (Dict): Contains the overall statistics for the batch run.
                                       For example, the dictionary returned by batch_run.
        """
        # This method can be used to supplement the main summary report with more structured log entries
        # Alternatively, if the main summary report is sufficient, this method can be simplified or removed
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Batch Run Completed ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            f.write(f"Dataset: {batch_summary_data.get('dataset_name', 'N/A')}\n")
            f.write(f"Total Scenarios in Dataset: {batch_summary_data.get('scenarios_count', 0)}\n")
            f.write(f"Scenarios Attempted: {batch_summary_data.get('scenarios_attempted', 0)}\n")
            f.write(f"Successful Runs (Game Flow): {batch_summary_data.get('successful_runs', 0)}\n")
            f.write(f"Failed Runs (Execution Error): {batch_summary_data.get('failed_runs', 0)}\n")
            f.write(f"Valid Evaluations: {batch_summary_data.get('valid_evaluation_count', 0)}\n")
            if batch_summary_data.get('valid_evaluation_count', 0) > 0:
                count = batch_summary_data['valid_evaluation_count']
                avg_o = batch_summary_data['total_overall_score'] / count
                avg_l = batch_summary_data['total_logic_accuracy'] / count
                avg_d = batch_summary_data['total_details_accuracy'] / count
                avg_c = batch_summary_data['total_conclusion_match'] / count
                f.write(f"Avg Overall: {avg_o:.4f}, Avg Logic: {avg_l:.4f}, Avg Details: {avg_d:.4f}, Avg Conclusion: {avg_c:.4f}\n")
            f.write("-" * 150 + "\n\n")
        
        # Ensure that if there's an unlogged periodic summary at the end of the batch, it gets logged
        if self.processed_scenarios_count % self.log_summary_interval != 0 and self.cumulative_eval_metrics["count"] > 0:
             # Check if there is new evaluation data since the last log
             if self.processed_scenarios_count > (self.processed_scenarios_count // self.log_summary_interval) * self.log_summary_interval:
                 self.log_periodic_summary()


    def get_processed_scenarios(self, dataset_name_filter: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Reads the list of processed scenarios from the log file, for resuming runs.
        Returns a dictionary where keys are dataset names and values are lists of successfully processed scenario titles in that dataset.
        Note: This is more complex than directly checking result files but can serve as a supplementary method.
              The current system uses result file checking to resume; this method can be used for validation or more granular control.
        """
        processed = {}
        if not self.log_file_path.exists():
            return processed
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Skip the header
                for _ in range(3): # Header, separator, blank line
                    next(f, None) 
                
                for line in f:
                    if line.startswith("---") or line.strip() == "": # Skip separators and blank lines
                        continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4: # Timestamp | Dataset | Scenario Title | Status | ...
                        dataset_name = parts[1]
                        title = parts[2]
                        status = parts[3].lower() # convert to lowercase for comparison
                        
                        if dataset_name_filter and dataset_name != dataset_name_filter:
                            continue

                        # Define which statuses count as "processed and do not need a re-run"
                        # e.g., "success", "failure" (a failure in the game's logic, but the program ran successfully)
                        # Scenarios with "error" status need to be re-run
                        if status in ["success", "failure"]:
                            if dataset_name not in processed:
                                processed[dataset_name] = []
                            if title not in processed[dataset_name]: # Avoid adding duplicates
                                processed[dataset_name].append(title)
        except Exception as e:
            print(f"Error reading log file {self.log_file_path}: {e}")
        return processed