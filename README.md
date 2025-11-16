# TurtleSoup-Bench

`TurtleSoup-Bench` is an automated benchmark framework designed to evaluate the questioning, reasoning, and summarization capabilities of Large Language Models (LLMs) using "Turtle Soup" (also known as situation puzzles).

The framework simulates the entire gameplay loop of a Turtle Soup puzzle by using AI agents to represent the "Questioner," "Responder," and "Evaluator," allowing for the quantitative scoring of a model's performance as the "Questioner."

## Core Architecture

The project uses a multi-agent system coordinated by the `TurtleSoupEngine` class in `turtle_system.py`. The system consists of three key components:

1.  **Questioner** (`Questioner.py`)
    * **Role**: The "Guesser" in the game.
    * **Task**: Analyzes the initial story "surface" and the conversation history to dynamically generate yes/no/irrelevant questions. It attempts to extract key clues from the answers and produces a final summary of the story's solution.

2.  **Responder** (`Responder.py`)
    * **Role**: The "Host" or "Knower."
    * **Task**: Accesses the ground truth solution (`bottom`) and tips (`tips`). It answers the Questioner's queries with "yes," "no," or "unknown" and identifies if a question has hit upon a "Key Clue."

3.  **Evaluator** (`Evaluator.py`)
    * **Role**: The "Judge."
    * **Task**: At the end of the game, it receives the `Questioner`'s final summary and compares it to the `true_solution`. It provides quantitative scores for **Logic**, **Details**, and **Conclusion Match** to evaluate the Questioner's accuracy.

## Datasets

The benchmark's puzzle data is located in the `data/` directory, providing a rich set of puzzles in both **English (`en`)** and **Chinese (`zh`)**.

The datasets are categorized by the core logic of the puzzles (as defined in `tools.py`):

* `Clever_Logic.json`
* `Crime_Thriller.json`
* `Constant_Change.json` (Corresponds to "Worldly Vicissitudes Type")
* `Mind_Game.json`
* `Supernatural_Fantasy.json`
* `Original_Data.json`

## Quick Start

### 1. Setup Environment

Ensure you have Python installed. First, clone the project, then install all required third-party libraries.

```bash
# It is recommended to use a Python virtual environment
pip install -r requirements.txt
````

The `requirements.txt` file should contain the following based on the project's imports:

```text
tqdm
openai
tenacity
numpy
google-generativeai
```

### 2\. Configure Models (Important)

This project relies on a `config.ini` file to manage and authenticate with all LLM APIs. You must configure this file to run the benchmark.

1.  Open `Code/config.ini`.
2.  Add a section (e.g., `[my-model-name]`) for each model you intend to test.
3.  In each section, provide the `model` name, `base_url` for the API, and your `api_key`.

**`config.ini` Example:**

```ini
[deepseek-r1]
model = deepseek-r1
base_url = your_deepseek_api_url
api_key = your_deepseek_api_key

[gpt-4o]
model = gpt-4o
base_url = your_openai_api_url
api_key = your_openai_api_key

[gemini-2.5-flash]
model = gemini-2.5-flash-preview-05-20
base_url = your_gemini_api_url
api_key = your_gemini_api_key
```

*Note: The code includes specific formatting adaptations for `o1` and `gemini` series models.*

### 3\. Run the Benchmark

Once configured, run the benchmark using the `turtle_system.py` script.

You must specify one or more `(Questioner, Responder)` model pairs to test using the `--model_pairs_json` argument.

**Example 1 (English, 5 stories from Crime\_Thriller):**

```bash
# Make sure to run from the 'Code' directory
python turtle_system.py \
    --language en \
    --story_path Crime_Thriller.json \
    --model_pairs_json '[["gpt-4o", "gpt-4o"]]' \
    --evaluator_model deepseek-r1 \
    --limit 5
```

**Example 2 (English, all stories from two categories, two model pairs):**

```bash
python turtle_system.py \
    --language en \
    --story_path Crime_Thriller.json Mind_Game.json \
    --model_pairs_json '[["gpt-4o", "gpt-4o"], ["deepseek-chat", "deepseek-chat"]]'
```

## Command-Line Arguments

Key arguments for `turtle_system.py`:

  * `--model_pairs_json` (Required): A JSON-formatted string defining one or more model pairs `[("Q_model_key", "R_model_key"), ...]`. The keys must match sections in `config.ini`.
  * `--story_path` (Required): One or more dataset filenames (e.g., `Crime_Thriller.json`) located in `data/[language]/`.
  * `--evaluator_model`: Specifies the "Judge" model (must match a `config.ini` section). (Default: `deepseek-r1`)
  * `--language`: The language to test (`en` or `zh`). (Default: `en`)
  * `--max_questions`: The maximum number of questions the Questioner can ask per story. (Default: 30)
  * `--limit`: Limits the number of stories to run from each dataset (for quick testing). (Default: None, runs all)
  * `--save_dir`: The root directory to save detailed results (JSON) and summary reports (TXT). (Default: `game_results`)
  * `--force_rerun`: Forces the script to re-run stories that already have a result file, otherwise, they are skipped.

## Output Structure

Results are saved to the directory specified by `--save_dir`, organized as follows:

```
[save_dir]/
│
└── [dataset_name]/  (e.g., Clever_Logic)
    │
    └── Q-[questioner_model]_R-[responder_model]/  (e.g., Q-gpt-4o_R-gpt-4o)
        │
        ├── [story_title]_[...models...].json  (Detailed JSON result for one story)
        ├── [story_title_2]_[...models...].json
        ├── run_log_[...].txt                   (Raw runtime log)
        └── summary_DS-[...].txt                (Final evaluation summary for this dataset)
```

  * **`.json` files**: Contain the full game metadata, complete conversation history, and the detailed evaluation scores for a single story.
  * **`.txt` summary file**: Provides aggregate statistics for the model pair on that dataset, including average scores and success rates.
