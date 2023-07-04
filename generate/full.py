import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from scripts.prepare_tasks import generate_prompt
from generate.base import generate
from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.model import Block
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization


def main(
    prompt: str = """<human>: You are a helpful assistant who can break incoming main task into an immediate sub task to complete the main task. You should use the observations and actions made till that point to come up with the next immediate sub task. And the next immediate sub task must fall into one of the domains below.

Domain 1 : movies_showtimes
All the tasks related to searching for movies, movie’s showtimes, etc falls in this domain. 

Domain 2 : movies_buy_tickets
All the tasks related to booking a movie ticket, searching for the availability of movie tickets, etc falls in this domain. 

Domain 3 : calendar_show_event
All the tasks related to enquiring about a person's availablility, information about someone’s free times, etc falls in this domain. 

Domain 4 : END
If the task is already completed with the previous sub tasks as given in the observation and actions made so far, then it should belong to this domain.

Given a main task and the previous observations and actions made so far, you have to come up with the next immediate sub task in order to complete the main task such that the sub task belongs to one of the domains above. If there are no more sub tasks required to complete the main task, then just reply with ‘(Domain: End)’. The subtask should always end with ‘###’.

Maintask: """,
    input: str = " Check for movie shows for John Wick 4  \nObservations and actions made Till now : \n <bot>:",
    finetuned_path: Path = Path("out/full/task/epoch100/lit_model_finetuned.pth"),
    checkpoint_dir: Path = Path(f"checkpoints/tiiuae/falcon-7b"),
    quantize: Literal["llm.int8", "gptq.int4"] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: str = "bf16-true",
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT model.
    See `finetune/full.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            `finetune/full.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    if strategy == "fsdp":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    if quantize is not None:
        # TODO: we need to clean-up the logic for quantizing the finetuned models and loading them after
        raise NotImplementedError
    checkpoint_path = finetuned_path

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.time()
    with lazy_load(finetuned_path) as checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=quantize is None)
    fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir)
    # sample = {"instruction": prompt, "input": input}
    # prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt + input, device=model.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(y)
    output = output.split("<bot>: ")[1].split("###")[0].strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
