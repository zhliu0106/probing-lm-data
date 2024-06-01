import torch
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    OPTForCausalLM,
    GPT2LMHeadModel,
    LlamaForCausalLM,
)


torch.set_grad_enabled(False)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROMPT = """Here is a statement:

[TEXT]

Is the above statement correct? Answer: """


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description="Generate activations")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--output_dir", default="./acts")
    args = parser.parse_args()
    return args


def read_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in tqdm(f)]
    new_data = []
    for d in data:
        new_data.append(
            {
                "text": d["text"],
                "label": d["label"],
            }
        )
    return new_data


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs


def get_acts(statements, tokenizer, model, layers):
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()

        if isinstance(model, LlamaForCausalLM):
            handle = model.model.layers[layer].register_forward_hook(hook)
        elif isinstance(model, GPTNeoXForCausalLM):
            handle = model.gpt_neox.layers[layer].register_forward_hook(hook)
        elif isinstance(model, OPTForCausalLM):
            handle = model.model.decoder.layers[layer].register_forward_hook(hook)
        elif isinstance(model, GPT2LMHeadModel):
            handle = model.transformer.h[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)

    # get activations
    acts = {layer: [] for layer in layers}
    for statement in statements:
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(model.device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])

    for layer, act in acts.items():
        acts[layer] = torch.stack(act).float()

    # remove hooks
    for handle in handles:
        handle.remove()

    return acts


if __name__ == "__main__":
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, return_dict=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.config.use_cache = True

    data = read_jsonl(args.dataset_path)

    model_name = args.model_path.split("/")[-1]
    save_dir = os.path.join(args.output_dir, model_name, args.dataset)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    statements = [PROMPT.replace("[TEXT]", ex["text"]) for ex in data]

    if isinstance(model, LlamaForCausalLM):
        layers = list(range(len(model.model.layers)))
    elif isinstance(model, GPTNeoXForCausalLM):
        layers = list(range(len(model.gpt_neox.layers)))
    elif isinstance(model, OPTForCausalLM):
        layers = list(range(len(model.model.decoder.layers)))
    elif isinstance(model, GPT2LMHeadModel):
        layers = list(range(len(model.transformer.h)))

    for idx in tqdm(range(0, len(statements), 25)):
        acts = get_acts(statements[idx : idx + 25], tokenizer, model, layers)
        for layer, act in acts.items():
            torch.save(act, os.path.join(save_dir, f"layer_{layer}_{idx}.pt"))
