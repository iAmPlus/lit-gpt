"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json, csv
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer

DATA_FILE_NAME = "iamai_brain_movies_lit.csv"
DESTINATION_PATH = Path("data/task")
CHECKPOINT_DIR = Path("checkpoints/tiiuae/falcon-7b-instruct")
TEST_SPLIT_SIZE = 5
IGNORE_INDEX = -1
MASK_INPUTS = False  # as in alpaca-lora
SEED = 42


def prepare(
    destination_path: Path = DESTINATION_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    test_split_size: int = TEST_SPLIT_SIZE,
    seed: int = SEED,
    mask_inputs: bool = MASK_INPUTS,
    data_file_name: str = DATA_FILE_NAME,    
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json", "r") as file:
        config = json.load(file)
        #max_seq_length = config["block_size"]
        max_seq_length = 1024

    destination_path.mkdir(parents=True, exist_ok=True)
    data_file_path = destination_path / data_file_name
    # print("Loading data file...")
    # download_if_missing(data_file_path, data_file_url)
    
    print(f"Reading file {data_file_path}")

    data = {}
    with open(data_file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [line for line in reader]

    print(data[0])
    print(len(data))
    # with open(data_file_path, "r", encoding="utf-8") as file:
    #     data = file.readlines()
    #     data = [json.loads(line) for line in data]


    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, lengths=(train_split_size, test_split_size), generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(file_url).text)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool = MASK_INPUTS,
    ignore_index: int = IGNORE_INDEX,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt_task(example):
    return(
        f"<human>: You are a helpful assistant who can break incoming main task into an immediate sub task to complete the main task. You should use the observations and actions made till that point to come up with the next immediate sub task. And the next immediate sub task must fall into one of the domains below.\n\nDomain 1 : movies_showtimes\nAll the tasks related to searching for movies, movie’s showtimes, etc falls in this domain. \n\nDomain 2 : movies_buy_tickets\nAll the tasks related to booking a movie ticket, searching for the availability of movie tickets, etc falls in this domain. \n\nDomain 3 : calendar_show_event\nAll the tasks related to enquiring about a person's availablility, information about someone’s free times, etc falls in this domain. \n\nDomain 4 : END\nIf the task is already completed with the previous sub tasks as given in the observation and actions made so far, then it should belong to this domain.\n\nGiven a main task and the previous observations and actions made so far, you have to come up with the next immediate sub task in order to complete the main task such that the sub task belongs to one of the domains above. If there are no more sub tasks required to complete the main task, then just reply with '(Domain: End)'. The subtask should always end with '###'.\n\nMainTask: {example['input']}\nObservations and actions made Till now : \n\n <bot>: "
    )

def generate_prompt(example):
    return (       
        f"{example['input']}"
    )
if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
