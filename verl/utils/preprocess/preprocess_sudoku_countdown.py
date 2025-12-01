# Convert Sudoku and Countdown datasets to JSON format expected by preprocess.py
import argparse
import os
import pandas as pd
import json
from src.utils import *
from tqdm import tqdm


SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where "0" represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="data/source", type=str, required=False, help="Output directory for processed files")
parser.add_argument("--n_few_shots", default=8, type=int, required=False, help="Number of few-shot examples to include")
args = parser.parse_args()


def generate_sudoku_few_shot_examples(n_examples=8):
    """Generate few-shot examples for Sudoku"""
    examples = [
        {"input": "0042\n0031\n0420\n3214", "output": "1342\n2431\n4123\n3214"},
        {"input": "1034\n3412\n0320\n0143", "output": "1234\n3412\n4321\n2143"},
        {"input": "3421\n0234\n0043\n4302", "output": "3421\n1234\n2143\n4312"},
        {"input": "0132\n2310\n3021\n0243", "output": "4132\n2314\n3421\n1243"},
        {"input": "2430\n0342\n4120\n0214", "output": "2431\n1342\n4123\n3214"},
        {"input": "1204\n3412\n0320\n0143", "output": "1234\n3412\n4321\n2143"},
        {"input": "0042\n0031\n0420\n3214", "output": "1342\n2431\n4123\n3214"},
        {"input": "1034\n3412\n0320\n0143", "output": "1234\n3412\n4321\n2143"}
    ]
    return examples[:n_examples]


def build_sudoku_prompt_with_examples(puzzle, n_few_shots=8):
    """Build Sudoku prompt with few-shot examples"""
    examples = generate_sudoku_few_shot_examples(n_few_shots)
    
    template = "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that each column, each row, and each of the four 2x2 subgrids that compose the grid contains all of the digits from 1 to 4.\n\n"
    template += "\n\n".join([f"Input:\n{i['input']}\nOutput:\n{i['output']}" for i in examples]) \
        + f"\n\nInput:\n{puzzle}\nOutput:\n "
    
    return template


def preprocess_sudoku_dataset(input_path, output_path, split):
    # Ensure puzzle and solution are loaded as strings to avoid leading zeros being removed
    df = pd.read_csv(input_path, dtype=str)
    dataset = df.to_dict("records")
    
    # Convert to the format expected by preprocess.py
    processed_data = []
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split} sudoku")):
        puzzle = example["Puzzle"]
        solution = example["Solution"]
        
        # # Add \n to 16-character strings to make them 4x4 grid format
        # puzzle = f"{puzzle[:4]}\n{puzzle[4:8]}\n{puzzle[8:12]}\n{puzzle[12:16]}"
        # solution = f"{solution[:4]}\n{solution[4:8]}\n{solution[8:12]}\n{solution[12:16]}"
        
        # question = build_sudoku_prompt_with_examples(puzzle, args.n_few_shots)
        question = f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {puzzle}\n"
        
        data_item = {
            "problem": question,
            "answer": solution,  # solution is a string
            "puzzle": puzzle,    # puzzle is a string
        }
        processed_data.append(data_item)
    print(processed_data[0])
    
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_path)
    
    print(f"Sudoku {split} total rows: {len(processed_data)}")
    print(f"Saved to: {output_path}")


def preprocess_countdown_dataset(input_path, output_path, split):
    if input_path.endswith(".parquet"):  # Train data
        df = pd.read_parquet(input_path)
        # Ensure nums column is a Python list instead of numpy array
        df["nums"] = df["nums"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
        dataset = df.to_dict("records")
    elif input_path.endswith(".jsonl"):  # Test data
        dataset = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    # Convert to the format expected by preprocess.py
    processed_data = []
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split} countdown")):
        if "target" in example and "nums" in example:  # Train data
            target = example["target"]
            nums = example["nums"]
        elif "input" in example and "output" in example:  # Test data
            nums_str = example["input"]  # 30,100,93
            target = example["output"]
            nums = [int(x.strip()) for x in nums_str.split(",")]
        else:
            continue
        
        question = f"{SYSTEM_PROMPT}\nUsing only the numbers {nums}, create an arithmetic expression that evaluates to exactly {target}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
        
        data_item = {
            "problem": question,
            "answer": str(target),  # target is a string
            "numbers": json.dumps(nums)  # Convert numbers to string format
        }
        processed_data.append(data_item)
    print(processed_data[0])
    
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_path)
    
    print(f"Countdown {split} total rows: {len(processed_data)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    os.makedirs(f"{args.output_dir}/train", exist_ok=True)
    os.makedirs(f"{args.output_dir}/test", exist_ok=True)
    
    # Sudoku train data
    sudoku_train_input = "data/source/train/4x4_sudoku_unique_puzzles.csv"
    sudoku_train_output = f"{args.output_dir}/train/sudoku.parquet"
    preprocess_sudoku_dataset(sudoku_train_input, sudoku_train_output, "train")
    
    # Sudoku test data
    sudoku_test_input = "data/source/test/4x4_test_sudoku.csv"
    sudoku_test_output = f"{args.output_dir}/test/sudoku.parquet"
    preprocess_sudoku_dataset(sudoku_test_input, sudoku_test_output, "test")
    
    # Countdown train data
    countdown_train_input = "data/source/train/countdown_3to4.parquet"
    countdown_train_output = f"{args.output_dir}/train/countdown.parquet"
    preprocess_countdown_dataset(countdown_train_input, countdown_train_output, "train")
    
    # Countdown test data
    countdown_test_input = "data/source/test/countdown_cd3_test.jsonl"
    countdown_test_output = f"{args.output_dir}/test/countdown.parquet"
    preprocess_countdown_dataset(countdown_test_input, countdown_test_output, "test")
