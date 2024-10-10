from typing import List

import torch
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoTokenizer

SOURCE_PREFIX = "<|source|>"
CAND1_PREFIX = "<|candidate1|>"
CAND2_PREFIX = "<|candidate2|>"


def tokenize_pair(tokenizer, sources: List[str], candidate1s: List[str], candidate2s: List[str], source_max_length=1224, candidate_max_length=412):
    ids = []
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(SOURCE_PREFIX + sources[i], max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(CAND1_PREFIX + candidate1s[i], max_length=candidate_max_length, truncation=True)
        candidate2_ids = tokenizer.encode(CAND2_PREFIX + candidate2s[i], max_length=candidate_max_length, truncation=True)
        ids.append(source_ids + candidate1_ids + candidate2_ids)

    encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
    return encodings


def get_comparison_results(queries: List[str], first_responses: List[str], second_responses: List[str], batch_size: int, cache_dir: str = None,
                           device=torch.device("cpu"), logger=None):
    pairrm = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf",
                                             cache_dir=cache_dir,
                                             device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf", cache_dir=cache_dir, trust_remote_code=True)

    encodings = tokenize_pair(tokenizer, queries, first_responses, second_responses)

    num_inputs = len(queries)
    num_batches = num_inputs // batch_size
    if num_inputs % batch_size != 0:
        num_batches += 1

    comparison_results = []
    for i in range(0, num_inputs, batch_size):
        if logger is not None:
            logger.info(f"Generating PairRM rankings for batch {i // batch_size + 1} / {num_batches}")

        batch_encodings = {k: v[i:i + batch_size].to(pairrm.device) for k, v in encodings.items()}
        outputs = pairrm(**batch_encodings)
        comparison_results.extend((outputs.logits > 0).tolist())

    return comparison_results
