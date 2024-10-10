import json
import os
from pathlib import Path

import jsonlines
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_and_model(model_name: str, load_model_checkpoint_from: str = "", is_lora_checkpoint: bool = False,
                             cache_dir: str = None, device=torch.device("cpu")):
    if not is_lora_checkpoint or not load_model_checkpoint_from:
        load_model_from = load_model_checkpoint_from if load_model_checkpoint_from else model_name
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=load_model_from,
            cache_dir=cache_dir,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model=model, model_id=load_model_checkpoint_from)
        model = model.merge_and_unload()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    return tokenizer, model


def load_jsonl(file_path: str):
    if not os.path.exists(file_path):
        return None

    json_list = []
    with jsonlines.open(file_path) as f:
        for line in f.iter():
            json_list.append(line)

    return json_list


def load_json(experiment_dir_path: Path, file_name: str):
    file_path = experiment_dir_path.joinpath(file_name)

    if not file_path.exists():
        return None

    with open(file_path) as f:
        return json.load(f)
