# Filter out problems that have both correct and incorrect answers in K sampling, remove pred and export parquet.
import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import json
from src.utils import load_jsonl


def _base_uid(unique_id: str) -> str:
    # Extract sampling suffixes like `{id}_0`
    m = re.match(r"^(.*)_\d+$", unique_id)
    if m:
        return m.group(1)
    return unique_id


def preprocess_dataset(dataset_name: str, baseline: str, sample_count: int, input_dir="data/correct", output_dir="preprocessed", max_file_size_gb: float = 1.5) -> None:
    input_file = os.path.join(input_dir, dataset_name, f"{baseline}-K{sample_count}.jsonl")
    samples = load_jsonl(input_file)

    # Load original data to get answer field (tests field for code tasks)
    uid_to_base: Dict[str, dict] = {}
    suffix = 1
    while True:
        preprocessed_file = f"data/preprocessed/{dataset_name}_{suffix}.parquet"
        if os.path.exists(preprocessed_file):
            df = pd.read_parquet(preprocessed_file)
            df = df.to_dict(orient="records")
            for record in df:
                uid_to_base[record["extra_info"]["index"]] = record
            suffix += 1
        else:
            break
    
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        grouped[_base_uid(sample["extra_info"]["index"])].append(sample)

    parquet_records: List[dict] = []
    parquet_index = 0
    for uid, items in grouped.items():
        labels = {int(it.get("is_correct", 0)) for it in items}
        if labels == {0, 1}:
            base = uid_to_base[uid]
            
            problem = base.get("problem") or items[0].get("problem")
            task = base["extra_info"]["task"]
            ground_truth = base["reward_model"]["ground_truth"]
            data_source = dataset_name.split("/", 1)[1] if "/" in dataset_name else dataset_name
            split = dataset_name.split("/", 1)[0] if "/" in dataset_name else "train"
            row = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": problem
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": split,
                    "index": parquet_index,
                    "task": task,
                }
            }

            # Pass metadata if it exists
            metadata = base.get("metadata")
            if metadata is not None and metadata != {}:
                row["extra_info"]["metadata"] = metadata
            parquet_records.append(row)
            parquet_index += 1

    os.makedirs(output_dir, exist_ok=True)
    parquet_base_path = os.path.join(output_dir, f"{dataset_name}-K{sample_count}.parquet")

    # Split into parquet files based on max_file_size
    max_file_size = max_file_size_gb * 1024 * 1024 * 1024
    file_idx = 1
    acc_size = 0
    chunk: List[dict] = []
    for rec in parquet_records:
        row_size = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        acc_size += row_size
        chunk.append(rec)
        if acc_size >= max_file_size:
            df = pd.DataFrame.from_records(chunk)
            out_path = f"{parquet_base_path.replace('.parquet', '')}_{file_idx}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"Write {out_path}, rows: {len(df)}")
            file_idx += 1
            acc_size = 0
            chunk = []
    if chunk:
        df = pd.DataFrame.from_records(chunk)
        out_path = f"{parquet_base_path.replace('.parquet', '')}_{file_idx}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Write {out_path}, rows: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--dataset_names", type=str, nargs="+", required=True)
    parser.add_argument("--dataset_sample_counts", type=int, nargs="+", required=True)
    parser.add_argument("--input_dir", type=str, default="data/correct")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed")
    parser.add_argument("--max_file_size_gb", type=float, default=1.5, help="Maximum size of a parquet file (GB)")
    args = parser.parse_args()

    for idx, dataset_name in enumerate(args.dataset_names):
        preprocess_dataset(dataset_name=dataset_name, baseline=args.baseline, sample_count=args.dataset_sample_counts[idx], input_dir=args.input_dir, output_dir=args.output_dir, max_file_size_gb=args.max_file_size_gb)
