import argparse
import os
import json
import asyncio
from datetime import datetime
import re
from typing import Counter, Dict, List, Optional
from configparser import ConfigParser
from tqdm import tqdm
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from Questioner import Questioner
from Responder import Responder
from Evaluator import Evaluator

from tools import *
import time

MAX_CONCURRENT_SCENARIOS = 1

class TurtleSoupEngine:
    """
    Core class for the Turtle Soup game engine.
    Responsible for coordinating gameplay, interacting with large models, managing state, and saving results.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initializes the game engine.
        Args:
            args: The parsed command-line arguments object.
        """
        self.args = args
        self.language = args.language
        self.max_questions = args.max_questions

        self.questioner_model_key = args.questioner_model
        self.responder_model_key = args.responder_model
        self.evaluator_model_key = args.evaluator_model

        # Get the current story filename and use it to create a dataset-specific save directory
        current_story_file = args.story_path[0] if isinstance(args.story_path, list) else args.story_path
        dataset_name = os.path.splitext(os.path.basename(current_story_file))[0]
        
        # Construct the folder name for the model pair
        q_model_folder_name = self.questioner_model_key.replace("/", "-").replace(":", "-")
        r_model_folder_name = self.responder_model_key.replace("/", "-").replace(":", "-")
        model_combination_folder_name = f"Q-{q_model_folder_name}_R-{r_model_folder_name}"

        # New self.save_dir structure: args.save_dir / dataset_name / model_combination_folder_name
        self.save_dir = os.path.join(args.save_dir, dataset_name, model_combination_folder_name)
        os.makedirs(self.save_dir, exist_ok=True) # Ensure the innermost directory exists

        # Set allowed answer types based on language
        self.allowed_answers = ALLOWED_ANSWERS_ZH if self.language == "zh" else ALLOWED_ANSWERS_EN
        
        self.clients: Dict[str, AsyncOpenAI] = {}
        self.model_configs = load_config()
        self._init_clients()

        self.key_clue_records: List[Dict] = []
        self.blacklist: set = set()
        self.last_overall_summary: Dict = {}

        # Initialize core components
        self.questioner = Questioner(self.args, self.clients, self.model_configs, self.language,
                                     self.key_clue_records, self.blacklist)
        self.responder = Responder(self.args, self.clients, self.model_configs, self.language,
                                   self.allowed_answers, self.blacklist)
        self.evaluator = Evaluator(self.args, self.clients, self.model_configs, self.language)

    def _init_clients(self):
        model_keys_to_init = set([
            self.questioner_model_key, self.responder_model_key, self.evaluator_model_key
        ])
        for model_key in model_keys_to_init:
            cfg = self.model_configs.get(model_key)
            if not cfg:
                raise ValueError(f"Model (section) not configured in config.ini: {model_key}")

            actual_model_name_from_config = cfg.get('model', model_key)

            # if 'gemini' in actual_model_name_from_config.lower():
            #     self.clients[model_key] = {"provider": "google", "config": cfg}
            # else: 
            self.clients[model_key] = AsyncOpenAI(base_url=cfg['base_url'], api_key=cfg['api_key'])

    # def _init_clients(self):
    #     model_keys_to_init = set([
    #         self.questioner_model_key, self.responder_model_key, self.evaluator_model_key
    #     ])
    #     for model_key in model_keys_to_init:
    #         cfg = self.model_configs.get(model_key)
    #         if not cfg:
    #             raise ValueError(f"Model (section) not configured in config.ini: {model_key}")

    #         actual_model_name_from_config = cfg.get('model', model_key) 

    #         if 'gemini' in actual_model_name_from_config.lower():
    #             self.clients[model_key] = {
    #                 "provider": "google", 
    #                 "model_name": actual_model_name_from_config, 
    #                 "api_key": cfg['api_key'], 
    #                 "config": cfg 
    #             }
    #         else: 
    #             self.clients[model_key] = AsyncOpenAI(base_url=cfg['base_url'], api_key=cfg['api_key'])

    async def run_scenario(self, title: str, setup: str, solution: str, tips: List[str]) -> Dict:
        start_time = datetime.now()
        history: List[Dict] = []
        progress: Optional[tqdm] = None
        result: Optional[Dict] = None

        self.key_clue_records.clear()
        self.blacklist.clear()
        self.last_overall_summary = {}
        self.questioner.type_state = {
            "current_type": "unknown_type" if self.language == "zh" else "default",
            "confidence": 0.5, "last_checked": -1, "last_key_clues_count": 0
        }
        self.questioner.last_internal_summary = {}

        try:
            progress = tqdm(
                total=self.max_questions,
                desc=f"Scenario[{title[:12]}]",
                leave=False,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
            )

            for q_num in range(self.max_questions + 1):
                question_content = "SUMMARY_TRIGGER"
                if q_num < self.max_questions:
                    question_content = await self.questioner.generate_question(setup, history, q_num)
                
                if question_content == "SUMMARY_TRIGGER" or q_num == self.max_questions:
                    final_summary_dict = await self.questioner.generate_final_summary(setup, history, self.last_overall_summary)
                    self.last_overall_summary = final_summary_dict
                    history.append({"type": "conclusion", "content": final_summary_dict})
                    evaluation = await self.evaluator.evaluate(true_solution=solution, pred_summary=final_summary_dict)
                    status = "success" if evaluation.get("overall_score", 0) >= 0.5 else "failure"
                    actual_questions_asked = q_num
                    result = self._build_result(status, actual_questions_asked, history, title, setup, solution, tips, start_time)
                    result["evaluation"] = evaluation
                    self._save_result(result)
                    return result

                answer_content = await self.responder.generate_answer_and_clue(setup, solution, tips, question_content)
                history.append({"type": "question", "content": question_content})
                history.append({"type": "answer", "content": answer_content})
                
                key_clue_tag_zh = "<Key Clue>" # Formerly "<关键线索>"
                key_clue_tag_en = "<Key Clue>"
                if (self.language == "zh" and key_clue_tag_zh in answer_content) or \
                   (self.language == "en" and key_clue_tag_en in answer_content):
                    self.key_clue_records.append({"question": question_content, "answer": answer_content})
                
                if progress: progress.update(1)

                if not validate_answer(answer_content):
                    result = self._build_result("error", q_num + 1, history, title, setup, solution, tips, start_time,
                                                error_msg=f"Invalid answer format: {answer_content}")
                    self._save_result(result)
                    return result
            
            if result is None:
                print(f"Warning: Scenario '{title}' loop finished without creating a final result object.")
                result = self._build_result("error", self.max_questions, history, title, setup, solution, tips, start_time,
                                            error_msg="Scenario processing loop completed unexpectedly without generating a clear result.")
                self._save_result(result)
                return result
        except Exception as e:
            import traceback
            print(f"A critical error occurred while processing scenario '{title}': {e}")
            traceback.print_exc()
            result = self._build_result("error", 0, history, title, setup, solution, tips, start_time, error_msg=str(e))
            self._save_result(result)
            return result 
        finally:
            if progress and not progress.disable:
                if progress.total is not None and progress.n < progress.total:
                    progress.update(progress.total - progress.n)
                progress.close() 
        
        if result is None:
            print(f"Critical Error: result is still None for scenario '{title}' after finally block execution.")
            result = self._build_result("error", 0, history, title, setup, solution, tips, start_time,
                                        error_msg="Scenario processing flow was abnormal; failed to generate a result.")
            self._save_result(result)
        return result

    def _build_result(self, status: str, q_count: int, history: List[Dict],
                      title: str, setup: str, solution: str, tips: List[str],
                      start_time: datetime, error_msg: str = None) -> Dict:
        q_model_name = getattr(self.args, 'questioner_model', 'unknown_q_model')
        r_model_name = getattr(self.args, 'responder_model', 'unknown_r_model')
        e_model_name = getattr(self.args, 'evaluator_model', 'unknown_e_model')
        return {
            "metadata": {
                "title": title, "status": status, "questions_used": q_count,
                "duration_sec": round((datetime.now() - start_time).total_seconds(), 1),
                "error": error_msg if status == "error" else None,
                "questioner_model": q_model_name, "responder_model": r_model_name,
                "evaluator_model": e_model_name, "language": self.language,
                "max_questions_setting": self.max_questions
            },
            "content": {
                "surface": setup, "bottom_line": solution, "tips_provided": tips,
                "conversation": [{"turn": i // 2 + 1, "type": msg["type"], "content": msg["content"]}
                                 for i, msg in enumerate(history)]
            }}

    def _save_result(self, result: Dict):
        meta = result.get("metadata", {})
        title_sanitized = re.sub(r'[\\/*?:"<>|]', "", meta.get("title", "unknown_title"))[:100]
        q_model_s = meta.get("questioner_model", "unknownQ").replace("/", "-").replace(":", "-")
        r_model_s = meta.get("responder_model", "unknownR").replace("/", "-").replace(":", "-")
        e_model_s = meta.get("evaluator_model", "unknownE").replace("/", "-").replace(":", "-")
        max_q_s = meta.get("max_questions_setting", self.max_questions)
        lang_s = meta.get("language", self.language)
        filename = f"{title_sanitized}_Q-{q_model_s}_R-{r_model_s}_E-{e_model_s}_Epoch{max_q_s}_{lang_s}.json"
        filepath = os.path.join(self.save_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving result to file {filepath}: {e}")

    async def close(self):
        clients_to_close = [client for client in self.clients.values() if client and hasattr(client, 'close')]
        if clients_to_close:
            await asyncio.gather(*(client.close() for client in clients_to_close))
        self.clients.clear()

async def run_single_scenario_wrapper(
    args_for_scenario: argparse.Namespace, 
    scenario_data: Dict, 
    semaphore: asyncio.Semaphore, 
    main_pbar: Optional[tqdm],
    logger: RunLogger # Receive Logger instance
):
    """Wrapper: Creates an engine instance, runs a single scenario, and logs the result upon completion."""
    engine = None
    result_from_scenario: Optional[Dict] = None 
    current_dataset_name = os.path.splitext(os.path.basename(args_for_scenario.story_path))[0]
    scenario_title_for_log = scenario_data.get('title', 'Unknown Title')

    try:
        engine = TurtleSoupEngine(args_for_scenario)
        async with semaphore:
            result_from_scenario = await engine.run_scenario(**scenario_data)
        
        # Real-time logging point
        if result_from_scenario:
            logger.log_scenario_result(current_dataset_name, result_from_scenario)
        
        return result_from_scenario
            
    except Exception as e:
        import traceback
        print(f"A critical error occurred in the wrapper for scenario {scenario_title_for_log}: {e}")
        traceback.print_exc()
        error_msg_wrapper = f"Wrapper error: {str(e)}"
        
        # Build error result
        if engine: 
            result_from_scenario = engine._build_result("error", 0, [], scenario_title_for_log, 
                                            scenario_data.get('setup',''), scenario_data.get('solution',''), 
                                            scenario_data.get('tips',[]), datetime.now(), error_msg=error_msg_wrapper)
        else: 
            q_model = getattr(args_for_scenario, 'questioner_model', 'unknownQ')
            r_model = getattr(args_for_scenario, 'responder_model', 'unknownR')
            e_model = getattr(args_for_scenario, 'evaluator_model', 'unknownE')
            lang = getattr(args_for_scenario, 'language', 'N/A')
            max_q = getattr(args_for_scenario, 'max_questions', 0)
            result_from_scenario = {"metadata": {"title": scenario_title_for_log, "status": "error", "error": error_msg_wrapper,
                                         "questioner_model": q_model, "responder_model": r_model,
                                         "evaluator_model": e_model, "language": lang,
                                         "max_questions_setting": max_q},
                                "content": scenario_data, "evaluation": {}}
        
        # Log this error result
        logger.log_scenario_result(current_dataset_name, result_from_scenario)
        return result_from_scenario # Return the error result
            
    finally:
        if engine:
            await engine.close()
        if main_pbar: 
            main_pbar.update(1)


async def batch_run(args_for_batch: argparse.Namespace, pbar_position: int = 0):
    """
    Batch run all (or a limited subset of) scenarios in a dataset.
    All outputs (JSON results, logs) will be saved to a directory specific to this dataset and model combination.
    """
    dataset_file_name = os.path.basename(args_for_batch.story_path)
    dataset_name_for_run = os.path.splitext(dataset_file_name)[0]

    # Construct the model pair's folder name for the path
    q_model_s_folder = args_for_batch.questioner_model.replace("/", "-").replace(":", "-")
    r_model_s_folder = args_for_batch.responder_model.replace("/", "-").replace(":", "-")
    model_pair_folder_name = f"Q-{q_model_s_folder}_R-{r_model_s_folder}"
    
    # Define the innermost specific directory for all outputs of this run (JSON results, logs, final TXT report)
    run_specific_output_dir = os.path.join(args_for_batch.save_dir, 
                                           dataset_name_for_run, 
                                           model_pair_folder_name)
    os.makedirs(run_specific_output_dir, exist_ok=True) # Ensure this directory exists
    
    # The Logger instance uses this innermost specific run directory
    logger = RunLogger(log_dir=run_specific_output_dir, 
                       run_id=dataset_name_for_run, 
                       model_pair_name=model_pair_folder_name) 

    # Load scenario data
    scenarios_path = os.path.join(args_for_batch.root_path, args_for_batch.language, args_for_batch.story_path)
    scenarios_all_in_file = load_scenarios(scenarios_path) 
    if not scenarios_all_in_file: # Handle file loading failure or empty content case
        print(f"Warning: Dataset file {scenarios_path} did not load any scenarios or is empty.")
        # Return an empty result structure indicating this situation
        return {
            "dataset_name": dataset_name_for_run, "args": vars(args_for_batch), 
            "scenarios_count": 0, "scenarios": [], "successful_runs": 0, "failed_runs": 0, 
            "error_message": f"No scenarios loaded from {scenarios_path}.",
            "valid_evaluation_count": 0, "total_logic_accuracy": 0.0, "total_details_accuracy": 0.0,
            "total_conclusion_match": 0.0, "total_overall_score": 0.0,
            "detailed_scene_evaluations": [f"Could not load any scenarios from {scenarios_path}."]
        }
    
    scenarios_to_consider_for_limit = [] # Store scenarios that need to be run or rerun based on their status
    skipped_due_to_success_failure = 0

    for scenario in scenarios_all_in_file:
        title_s = re.sub(r'[\\/*?:"<>|]', "", scenario.get("title", "unknown_title"))[:100]
        q_model_s_file = args_for_batch.questioner_model.replace("/", "-").replace(":", "-")
        r_model_s_file = args_for_batch.responder_model.replace("/", "-").replace(":", "-")
        e_model_s_file = args_for_batch.evaluator_model.replace("/", "-").replace(":", "-")
        max_q_s_file = args_for_batch.max_questions
        lang_s_file = args_for_batch.language
        
        expected_filename = f"{title_s}_Q-{q_model_s_file}_R-{r_model_s_file}_E-{e_model_s_file}_Epoch{max_q_s_file}_{lang_s_file}.json"
        expected_filepath = os.path.join(run_specific_output_dir, expected_filename)

        should_run_this_scenario = True # Default to needing a run

        if os.path.exists(expected_filepath) and not args_for_batch.force_rerun:
            try:
                with open(expected_filepath, 'r', encoding='utf-8') as f_check:
                    existing_data = json.load(f_check)
                existing_status = existing_data.get("metadata", {}).get("status", "").lower()
                
                if existing_status in ["success", "failure"]:
                    should_run_this_scenario = False 
                    skipped_due_to_success_failure += 1
                elif existing_status == "error":
                    print(f"Info: Record '{title_s}' (file: {expected_filename}) previously had status 'error', will re-run.")
                    should_run_this_scenario = True 
                else:
                    print(f"Info: Record '{title_s}' (file: {expected_filename}) has unknown status ('{existing_status}') or no status, will re-run.")
                    should_run_this_scenario = True
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read or parse existing record file {expected_filepath} ({e}), will re-run scenario '{title_s}'.")
                should_run_this_scenario = True
        elif args_for_batch.force_rerun:
            should_run_this_scenario = True 

        if should_run_this_scenario:
            scenarios_to_consider_for_limit.append(scenario)
    
    if skipped_due_to_success_failure > 0:
        print(f"Dataset [{dataset_name_for_run}]: Skipped {skipped_due_to_success_failure} scenarios with existing 'success' or 'failure' records.")

    # Apply the limit parameter to the filtered list
    scenarios_to_run_final = scenarios_to_consider_for_limit
    if args_for_batch.limit is not None and args_for_batch.limit > 0:
        if len(scenarios_to_consider_for_limit) > args_for_batch.limit:
            print(f"Dataset [{dataset_name_for_run}]: After filtering, {len(scenarios_to_consider_for_limit)} scenarios to run/re-run. Will run the first {args_for_batch.limit} due to the limit parameter.")
            scenarios_to_run_final = scenarios_to_consider_for_limit[:args_for_batch.limit]
    
    # Log initial batch information
    planned_titles = [s.get("title", "N/A") for s in scenarios_to_run_final]
    if scenarios_to_run_final:
        logger.log_initial_batch_info({dataset_file_name: planned_titles})
    else: 
        logger.log_initial_batch_info({dataset_file_name: ["No new scenarios or scenarios to re-run this time."]})


    if not scenarios_to_run_final: # If there are no scenarios to run in the end
        print(f"Dataset [{dataset_name_for_run}]: No new tasks or 'error' status scenarios to re-run this time.")
        return { 
            "dataset_name": dataset_name_for_run, "args": vars(args_for_batch), 
            "scenarios_count": len(scenarios_all_in_file), "scenarios": scenarios_all_in_file, 
            "successful_runs": 0, "failed_runs": 0,
            "valid_evaluation_count": 0, "total_logic_accuracy": 0.0, "total_details_accuracy": 0.0,
            "total_conclusion_match": 0.0, "total_overall_score": 0.0,
            "detailed_scene_evaluations": [f"All scenarios were skipped or no tasks were available."]
        }
    
    # Create a semaphore to control concurrency
    semaphore = asyncio.Semaphore(args_for_batch.max_concurrent_scenarios)
    print(f"Dataset [{dataset_name_for_run}]: A total of {len(scenarios_to_run_final)} new scenarios to process. Concurrency set to: {args_for_batch.max_concurrent_scenarios}")

    tasks = []
    # Create the main progress bar for the dataset
    with tqdm(
        total=len(scenarios_to_run_final), desc=f"Dataset [{dataset_name_for_run}]",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} Scenarios [{percentage:3.0f}%]",
        position=pbar_position, leave=True 
    ) as main_pbar:
        for scenario_data in scenarios_to_run_final:
            task = run_single_scenario_wrapper(
                args_for_batch, scenario_data, semaphore, main_pbar, logger 
            )
            tasks.append(task)
        results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

    # --- Result Aggregation ---
    total_logic_accuracy, total_details_accuracy, total_conclusion_match, total_overall_score = 0.0, 0.0, 0.0, 0.0
    valid_evaluation_count, successful_runs, failed_runs = 0, 0, 0
    detailed_scene_evaluations_for_report = [] 

    for i, result_item in enumerate(results_from_gather):
        current_scenario_data = scenarios_to_run_final[i] 
        scenario_title = current_scenario_data.get('title', f"Scenario {i+1}")
        scene_eval_line = f"Scenario '{scenario_title}': "

        if isinstance(result_item, Exception):
            failed_runs += 1
            scene_eval_line += f"Execution threw an exception: {type(result_item).__name__} - {str(result_item)}"
        elif result_item is None: 
            failed_runs += 1
            scene_eval_line += "Failed to process (wrapper returned None)."
        else: 
            metadata = result_item.get("metadata", {})
            if metadata.get("status") == "error":
                failed_runs += 1
                scene_eval_line += f"Reported error: {metadata.get('error', 'Unknown error')}"
            elif metadata.get("status") in ["success", "failure"]:
                successful_runs += 1
                evaluation_data = result_item.get("evaluation")
                if isinstance(evaluation_data, dict) and evaluation_data: 
                    logic_acc = evaluation_data.get("logic_accuracy")
                    details_acc = evaluation_data.get("details_accuracy")
                    conclusion_match_score = evaluation_data.get("conclusion_match")
                    overall_score_val = evaluation_data.get("overall_score")
                    if all(isinstance(val, (int, float)) for val in [logic_acc, details_acc, conclusion_match_score, overall_score_val]):
                        total_logic_accuracy += logic_acc; total_details_accuracy += details_acc
                        total_conclusion_match += conclusion_match_score; total_overall_score += overall_score_val
                        valid_evaluation_count += 1
                        scene_eval_line += (f"L:{logic_acc:.2f} D:{details_acc:.2f} C:{conclusion_match_score:.2f} O:{overall_score_val:.2f}")
                    else: scene_eval_line += "Evaluation data is incomplete or has an incorrect format."
                else: scene_eval_line += "No valid evaluation data found."
        detailed_scene_evaluations_for_report.append(scene_eval_line)
        
    batch_summary_data_for_log_and_return = {
        "dataset_name": dataset_name_for_run, "args": vars(args_for_batch), 
        "scenarios_count": len(scenarios_all_in_file), 
        "scenarios_attempted": len(scenarios_to_run_final), 
        "successful_runs": successful_runs, "failed_runs": failed_runs, 
        "valid_evaluation_count": valid_evaluation_count,
        "total_logic_accuracy": total_logic_accuracy, 
        "total_details_accuracy": total_details_accuracy,
        "total_conclusion_match": total_conclusion_match, 
        "total_overall_score": total_overall_score,
        "detailed_scene_evaluations": detailed_scene_evaluations_for_report, 
        "original_scenarios_list": scenarios_all_in_file 
    }
    logger.log_batch_completion(batch_summary_data_for_log_and_return)
    
    return batch_summary_data_for_log_and_return


async def main_async(main_args: argparse.Namespace):
    """
    Main asynchronous function to coordinate batch runs of different datasets with different model pairs,
    and to generate a final summary report.
    """
    datasets_to_run = main_args.story_path      
    model_pairs_to_run = main_args.model_pairs  

    pbar_row_increment_per_dataset = 1 
    current_pbar_start_row = 0 

    for q_model, r_model in model_pairs_to_run:
        print(f"\n{'='*25} Starting to process model combination {'='*25}")
        print(f"Questioner: '{q_model}', Responder: '{r_model}'")
        print(f"Will concurrently process the following datasets: {', '.join(datasets_to_run)}")
        print(f"{'='*70}\n")

        dataset_tasks = [] # Store the batch_run async task for each dataset
        # Inner loop: process all specified datasets for the current model pair
        for dataset_file in datasets_to_run: 
            args_for_dataset_run = argparse.Namespace(**vars(main_args)) 
            args_for_dataset_run.questioner_model = q_model
            args_for_dataset_run.responder_model = r_model
            args_for_dataset_run.evaluator_model = main_args.evaluator_model
            args_for_dataset_run.story_path = dataset_file 
            
            # Create a batch_run task and pass the starting row number for the main progress bar
            task = batch_run(args_for_dataset_run, pbar_position=current_pbar_start_row)
            dataset_tasks.append(task)
            
            # Calculate the new starting row number for the next dataset's progress bar to avoid overlap
            current_pbar_start_row += pbar_row_increment_per_dataset
        
        # Concurrently execute all batch_run tasks for the current model combination
        all_dataset_results = await asyncio.gather(*dataset_tasks, return_exceptions=True)
        
        print(f"\n\n{'='*20} Model combination ({q_model}/{r_model}) run complete, generating summary report {'='*20}")
        
        # Process the results of each dataset run and generate a TXT summary report
        for dataset_run_result in all_dataset_results: 
            if isinstance(dataset_run_result, Exception): 
                print(f"A dataset task failed (caught in gather): {dataset_run_result}")
                import traceback 
                if hasattr(dataset_run_result, '__traceback__') and dataset_run_result.__traceback__:
                    traceback.print_tb(dataset_run_result.__traceback__)
                continue
            if dataset_run_result is None: # batch_run should not return None, unless there is a logical error
                print(f"A dataset task returned None, skipping report generation.")
                continue
            
            # Get parameters from the result dictionary returned by batch_run, which are specific to that dataset run
            args_from_result_dict = dataset_run_result.get("args", {})
            if not args_from_result_dict:
                print("Warning: The result from batch_run is missing the 'args' field, cannot generate summary report for this dataset.")
                continue

            dataset_name_from_result = dataset_run_result.get("dataset_name", "unknown_dataset")
            # Get model names from the result's args to ensure consistency with the actual run
            q_model_s_report = args_from_result_dict.get('questioner_model','unknownQ').replace("/", "-").replace(":", "-")
            r_model_s_report = args_from_result_dict.get('responder_model','unknownR').replace("/", "-").replace(":", "-")
            e_model_s_report = args_from_result_dict.get('evaluator_model','unknownE').replace("/", "-").replace(":", "-")
            
            # Construct the model pair folder name consistent with batch_run and TurtleSoupEngine
            model_pair_folder_name_for_report = f"Q-{q_model_s_report}_R-{r_model_s_report}"
            
            # report_final_dir points to the innermost directory: global save_dir / dataset_name / model_combination_name
            report_final_dir = os.path.join(args_from_result_dict.get('save_dir', 'game_results'), 
                                              dataset_name_from_result, 
                                              model_pair_folder_name_for_report)
            os.makedirs(report_final_dir, exist_ok=True) # Re-ensure the directory exists

            # summary_filename itself is unchanged, but it will now be saved under report_final_dir
            summary_filename = f"summary_DS-{dataset_name_from_result}_Q-{q_model_s_report}_R-{r_model_s_report}_E-{e_model_s_report}.txt"
            summary_filepath = os.path.join(report_final_dir, summary_filename)

            try: # Attempt to write the summary report file
                with open(summary_filepath, 'a', encoding='utf-8') as f: # Use append mode 'a'
                    f.write(f"\n\n=== Batch Run Record ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
                    f.write(f"--- Model and Dataset Configuration ---\n")
                    f.write(f"Dataset File: {args_from_result_dict.get('story_path', 'N/A')}\n")
                    f.write(f"Questioner Model: {q_model_s_report}\n")
                    f.write(f"Responder Model: {r_model_s_report}\n")
                    f.write(f"Evaluator Model: {e_model_s_report}\n") # Add evaluator model to the report
                    f.write(f"Language: {args_from_result_dict.get('language', 'N/A')}\n")
                    f.write(f"Max Questions Setting: {args_from_result_dict.get('max_questions', 'N/A')}\n")
                    f.write(f"Scenario Limit: {args_from_result_dict.get('limit', 'None')}\n")
                    
                    f.write(f"\n--- List of Scenarios in This Batch Run (Total {dataset_run_result.get('scenarios_count',0)}) ---\n")
                    original_scenarios = dataset_run_result.get('original_scenarios_list', [])
                    if original_scenarios:
                        for idx, scenario_info in enumerate(original_scenarios):
                            f.write(f"{idx+1}. {scenario_info.get('title', f'Unknown Title Scenario_{idx+1}')}\n")
                    else: 
                        f.write("Could not load scenario information or dataset is empty.\n")
                    f.write("\n")

                    f.write(f"--- This Batch Run's Statistics ---\n")
                    f.write(f"Total Scenarios in Dataset: {dataset_run_result.get('scenarios_count',0)}\n")
                    f.write(f"Scenarios Attempted in this Run: {dataset_run_result.get('scenarios_attempted',0)}\n")
                    f.write(f"Scenarios with Successful Game Flow: {dataset_run_result.get('successful_runs',0)}\n")
                    f.write(f"Scenarios with Program Execution Errors: {dataset_run_result.get('failed_runs',0)}\n")
                    f.write(f"Scenarios with Valid Evaluation Data: {dataset_run_result.get('valid_evaluation_count',0)}\n\n")

                    valid_eval_count = dataset_run_result.get('valid_evaluation_count',0)
                    if valid_eval_count > 0:
                        avg_logic = dataset_run_result.get('total_logic_accuracy',0) / valid_eval_count
                        avg_details = dataset_run_result.get('total_details_accuracy',0) / valid_eval_count
                        avg_conclusion = dataset_run_result.get('total_conclusion_match',0) / valid_eval_count
                        avg_overall = dataset_run_result.get('total_overall_score',0) / valid_eval_count

                        f.write(f"--- Average Evaluation Metrics for This Batch Run (based on {valid_eval_count} valid evaluations) ---\n")
                        f.write(f"Average Logic Accuracy: {avg_logic:.4f}\n")
                        f.write(f"Average Details Accuracy: {avg_details:.4f}\n")
                        f.write(f"Average Conclusion Match: {avg_conclusion:.4f}\n")
                        f.write(f"Average Overall Score: {avg_overall:.4f}\n\n")
                    else:
                        f.write("No valid evaluation data found in this batch run to calculate average metrics.\n\n")
                    
                    f.write("--- Detailed Evaluation for Each Scenario in This Batch Run (from TXT report) ---\n") 
                    detailed_evals_for_report = dataset_run_result.get('detailed_scene_evaluations', [])
                    if detailed_evals_for_report:
                        for line in detailed_evals_for_report: 
                            f.write(line + "\n")
                    else:
                        f.write("No detailed scenario evaluation records.\n")
                    
                    f.write("=== End of Record ===\n")

                print(f"Summary report for dataset '{dataset_name_from_result}' (Q:{q_model_s_report}/R:{r_model_s_report}) has been appended to: {summary_filepath}")
            except IOError as e:
                print(f"\nError: Could not write summary to file {summary_filepath}. Error info: {e}")
            except Exception as e_report: # Capture other unknown errors during report generation
                print(f"\nAn unexpected error occurred while generating the summary report for dataset '{dataset_name_from_result}': {e_report}")
                import traceback
                traceback.print_exc()
        
        time.sleep(1) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turtle Soup Automatic Game and Evaluation System")
    # Model related arguments
    # parser.add_argument("--questioner_model", type=str, default="glm-4-flash", help="Model name for the Questioner (corresponds to a section in config.ini)")
    # parser.add_argument("--responder_model", type=str, default="glm-4-flash", help="Model name for the Responder")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-r1", help="Model name for the Evaluator")

    # Path related arguments
    parser.add_argument("--root_path", type=str, default="data", help="Root directory for datasets")
    
    # "Crime_Thriller.json", "Mind_Game.json", "Supernatural_Fantasy.json", "Constant_Change.json", "Clever_Logic.json"
    # "Original_Data.json"
    parser.add_argument("--story_path", nargs='+', default=["Original_Data.json"], help="Story filename or list of files relative to root_path/language/")
    parser.add_argument("--save_dir", type=str, default="game_results", help="Directory to save game results and logs")

    # Game configuration arguments
    parser.add_argument("--language", choices=["en", "zh"], default="en", help="Game language (en or zh)")
    parser.add_argument("--max_questions", type=int, default=30, help="Maximum number of questions allowed per scenario")
    
    # 82 60 31 58 69 100
    parser.add_argument("--limit", type=int, default=1, help="Limit the number of scenarios to process in each dataset")
    
    parser.add_argument("--force_rerun", action='store_true', default=False, help="Force re-running scenarios that already have results")
    parser.add_argument("--max_concurrent_scenarios", type=int, default=MAX_CONCURRENT_SCENARIOS, help="Maximum number of scenarios to run concurrently")

    # Questioner internal parameters
    parser.add_argument("--summary_interval", type=int, default=5, help="Interval of questions for the Questioner to perform internal summarization")
    parser.add_argument("--history_window_prompt", type=int, default=10, help="Number of historical conversation turns for the Questioner to reference when generating a question")
    parser.add_argument("--history_window_selection", type=int, default=4, help="Number of historical conversation turns for the Questioner to reference when selecting a question type")

    # Model pair configuration parameters
    default_model_pairs = [
        # ("glm-4-flash", "glm-4-flash"),
        # ("o3", "o3")
        # ("gpt-4o", "gpt-4o"),
        # ("qwen3-32b", "qwen3-32b"),
        # ("deepseek-chat", "deepseek-chat")
        # ("deepseek-r1", "deepseek-r1"),
        # ("llama3-8b-instruct", "llama3-8b-instruct"),
        # ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash-preview-05-20"),
        # ("claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219"),
        # ("o3-mini-2025-01-31", "o3-mini-2025-01-31")
    ] 
    
    parser.add_argument('--model_pairs_json', type=str, default=json.dumps(default_model_pairs), 
                        help='A JSON string representing the list of model pairs')
    
    cli_args = parser.parse_args() 
    # Parse the JSON string format of model pairs into a Python list
    cli_args.model_pairs = json.loads(cli_args.model_pairs_json)
    asyncio.run(main_async(cli_args))