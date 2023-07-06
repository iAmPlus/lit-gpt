import os
import json, csv
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from copy import deepcopy
from itertools import chain
from datasets import load_dataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DATASET_PATH = "./data/orca/orca_flan1m_gpt4.csv"
DESTINATION_PATH = Path("data/orca")
CHECKPOINT_DIR = Path("checkpoints/tiiuae/falcon-7b")
TEST_SPLIT_PERCENTAGE = 1
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42
BLOCK_SIZE = 2020
PREPROCESSING_NUM_WORKERS = 12

USER_TOKEN = "<human>:"
ASSISTANT_TOKEN = "<bot>:"
END_TOKEN = "###"

def find_sublist(main_list, sublist):
    indices = []
    len_sublist = len(sublist)
    for i in range(len(main_list)):
        if i+len_sublist <= len(main_list) and main_list[i:i+len_sublist] == sublist:
            indices.append(i)
    return indices

def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_percentage: int = TEST_SPLIT_PERCENTAGE,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    dataset_path: str = DATASET_PATH,    
    ignore_index: int = IGNORE_INDEX,
    preprocessing_num_workers: int = PREPROCESSING_NUM_WORKERS,
    block_size: int = BLOCK_SIZE,
) -> None:
    
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    # print("Loading data file...")
    # download_if_missing(data_file_path, data_file_url)
    
    print(f"Reading file {dataset_path}")
    data_files = {}
    data_files["train"] = dataset_path
    extension = dataset_path.split(".")[-1]
    assert extension in ["csv"], "`train_file` should be a csv file."
    
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{test_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:]",
        )

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    #print(f"train has {len(raw_datasets["train"]):,} samples")
    #print(f"test has {len(raw_datasets["validation"]):,} samples")

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return {"input_ids" : [tokenizer.encode(example) for example in examples[text_column_name]]}

    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    print("Tokenized dataset : ", tokenized_datasets)
    
    
    def group_texts(examples):
        user_token_ids = tokenizer.encode(USER_TOKEN).tolist()
        assistant_token_ids = tokenizer.encode(ASSISTANT_TOKEN).tolist()
        end_token_id = tokenizer.encode(END_TOKEN).item()
        pad_token_id = tokenizer.eos_id
        next_line_token_id = tokenizer.encode("\n").item()

        keys_list = list(examples.keys())

        concatenated_examples = {k: list(chain(*examples[k])) for k in keys_list}
        
        user_indices = find_sublist(concatenated_examples[keys_list[0]],user_token_ids)

        final_block_convs, temp_block_conv, prev_conv = [], [], []
        for idx1 in range(len(user_indices)):
            start_idx = user_indices[idx1]
            if idx1+1 < len(user_indices):
                end_idx = user_indices[idx1+1]
            else:
                end_idx = len(concatenated_examples[keys_list[0]])

            current_conv = concatenated_examples[keys_list[0]][start_idx:end_idx]

            if len(current_conv)>block_size:
                current_conv = current_conv[:block_size]
                current_conv[-2] = end_token_id
                current_conv[-1] = next_line_token_id

            if len(temp_block_conv)==0 and len(prev_conv)>0:
                assistant_index_list = find_sublist(prev_conv, assistant_token_ids)
                if(len(assistant_index_list)>0):
                    assistant_index = assistant_index_list[0]
                    if len(prev_conv[assistant_index:]) <= block_size/2:
                        temp_block_conv = deepcopy(prev_conv[assistant_index:])
                prev_conv.clear()
                
            if len(temp_block_conv) + len(current_conv) <= block_size:
                temp_block_conv.extend(deepcopy(current_conv))
                prev_conv = deepcopy(current_conv)
            else:
                while(len(temp_block_conv)<block_size):
                    temp_block_conv.append(pad_token_id)
                idx1 = idx1 - 1
            
            if len(temp_block_conv)==block_size:
                if len(prev_conv)>0:
                    final_block_convs.append(deepcopy(temp_block_conv))
    
                temp_block_conv.clear()

        result = {keys_list[0]:deepcopy(final_block_convs)}
        labels = deepcopy(result["input_ids"])
        result["labels"] = labels

        result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.int)
        result["labels"] = torch.tensor(result["labels"], dtype=torch.int)
        #print(result)
        return result
    

    lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    print("lm dataset : ", lm_datasets)

    
    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["validation"]

    print("Processing train split ...")
    train_set = [sample for sample in train_dataset]
    for i in range(len(train_set)):
        train_set[i]["input_ids"] = torch.tensor(train_set[i]["input_ids"], dtype=torch.int)
        train_set[i]["labels"] = torch.tensor(train_set[i]["labels"], dtype=torch.int)
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [sample for sample in test_dataset]
    for i in range(len(test_set)):
        test_set[i]["input_ids"] = torch.tensor(test_set[i]["input_ids"], dtype=torch.int)
        test_set[i]["labels"] = torch.tensor(test_set[i]["labels"], dtype=torch.int)
    torch.save(test_set, destination_path / "test.pt")
    


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)

def generate_prompt(example):
    return (       
        f"{example['input']}"
    )
if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
