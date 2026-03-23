#!/usr/bin/env python3
"""
BvB / Regime Collapse - Llama 2 robustness sweeps

Purpose
-------
Run the same regime-collapse alignment sweep on larger open-weight models that are
realistic on Kaggle via parameter-efficient fine-tuning (QLoRA/LoRA), rather than
full-weight fine-tuning.

Default model
-------------
- meta-llama/Llama-2-7b-hf

Notes
-----
- Llama 2 access requires a Hugging Face token with permission to the Meta model.
- On Kaggle, prefer NVIDIA T4 over P100 for bitsandbytes/4-bit workflows.
- The goal here is robustness sweeps, not exact numerical identity with the GPT-2
  results, since these runs use PEFT to fit larger models into Kaggle memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


DEFAULT_MODELS = ["meta-llama/Llama-2-7b-hf"]
DEFAULT_SEEDS = [0]
DEFAULT_STEPS = [150, 300]
DEFAULT_LAMBDAS = [0.0, 0.01, 0.025]
DEFAULT_BLOCK_CHARS = 1500
DEFAULT_LR = 2e-4
DEFAULT_BATCH = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_PROBE_BS = 4
DEFAULT_LOGGING_STEPS = 20
DEFAULT_MAX_LENGTH = 768


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def gpu_clean() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_latex(text: str) -> str:
    text = re.sub(r"\\begin\{[^}]*\}", " ", text)
    text = re.sub(r"\\end\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\$+", " ", text)
    text = re.sub(r"\\[\\&%#_^~]", " ", text)
    text = re.sub(r"%[^\n]*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_file(filename: str, search_roots: Sequence[str]) -> str:
    for search_root in search_roots:
        if not search_root:
            continue
        candidate = os.path.join(search_root, filename)
        if os.path.isfile(candidate):
            return candidate
        for root, _, files in os.walk(search_root):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(f"Cannot find {filename} under roots: {search_roots}")


def make_blocks(text: str, block_chars: int) -> Dataset:
    text = text.replace("\r\n", "\n")
    blocks = []
    for i in range(0, len(text), block_chars):
        blk = text[i:i + block_chars]
        if blk.strip():
            blocks.append({"text": blk})
    return Dataset.from_list(blocks)


def split_ds(ds: Dataset, n_train: int, n_eval: int, seed: int = 0) -> Tuple[Dataset, Dataset]:
    ds_shuffled = ds.shuffle(seed=seed)
    n_train = min(n_train, len(ds_shuffled))
    n_eval_actual = min(n_eval, len(ds_shuffled) - n_train)
    if n_eval_actual <= 0:
        raise ValueError(
            f"Not enough data: have {len(ds_shuffled)}, need {n_train} train + {n_eval} eval"
        )
    train = ds_shuffled.select(range(n_train))
    eval_ = ds_shuffled.select(range(n_train, n_train + n_eval_actual))
    return train, eval_


@dataclass
class PreparedData:
    train_langA: Dataset
    eval_langA: Dataset
    train_langB: Dataset
    eval_langB: Dataset


def prepare_language_data(input_roots: Sequence[str], block_chars: int = DEFAULT_BLOCK_CHARS) -> PreparedData:
    lang_a_path = find_file("regime_A.txt", input_roots)
    lang_b_path = find_file("regime_B S.txt", input_roots)
    lang_A = load_text(lang_a_path)
    lang_B = load_text(lang_b_path)

    dsA = make_blocks(lang_A, block_chars)
    dsB = make_blocks(lang_B, block_chars)
    train_langA, eval_langA = split_ds(dsA, n_train=600, n_eval=256, seed=0)
    train_langB, eval_langB = split_ds(dsB, n_train=40, n_eval=39, seed=0)
    return PreparedData(
        train_langA=train_langA,
        eval_langA=eval_langA,
        train_langB=train_langB,
        eval_langB=eval_langB,
    )


class TextTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dataset, tokenizer: AutoTokenizer, max_length: int):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        text = self.ds[idx]["text"]
        toks = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}



def maybe_set_hf_token_env(token: Optional[str]) -> None:
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token



def build_tokenizer(model_name: str, token: Optional[str] = None) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok



def build_qlora_model(model_name: str, token: Optional[str] = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model



def eval_loss_quick(model, dataset, collator, device: str, batch_size: int = 2) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            idx = range(i, min(i + batch_size, len(dataset)))
            batch = collator([dataset[j] for j in idx])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
    return float(np.mean(losses))



def _load_existing(csv_path: str) -> List[dict]:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).to_dict("records")
    return []



def _done_keyed(rows: Iterable[dict], key_cols: Sequence[str]) -> set:
    done = set()
    for r in rows:
        done.add(tuple(float(r[c]) if c == "lambda_align" else int(r[c]) if c in {"seed", "steps"} else r[c] for c in key_cols))
    return done



def run_large_model_sweep(
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    collator: DataCollatorForLanguageModeling,
    trainA,
    trainB,
    evalA,
    evalB,
    output_dir: str,
    csv_name: str,
    lambdas: Sequence[float],
    seeds: Sequence[int],
    steps_list: Sequence[int],
    learning_rate: float,
    train_batch_size: int,
    grad_accum: int,
    probe_bs: int,
    logging_steps: int,
    hf_token: Optional[str],
) -> pd.DataFrame:
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, csv_name)
    rows = _load_existing(csv_path)
    done = _done_keyed(rows, ["model", "lambda_align", "seed", "steps"]) if rows else set()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for lam in lambdas:
        for seed in seeds:
            set_all_seeds(seed)
            probe_ds = trainB.shuffle(seed=seed).select(range(min(probe_bs, len(trainB))))
            probe_batch = collator([probe_ds[i] for i in range(len(probe_ds))])

            for steps in steps_list:
                key = (model_name, float(lam), int(seed), int(steps))
                if key in done:
                    print(f"[skip] {key} already complete")
                    continue

                print("=" * 80)
                print(f"model={model_name} | λ={lam:.4f} | seed={seed} | steps={steps}")
                print("=" * 80)

                gpu_clean()
                model = build_qlora_model(model_name, hf_token)
                probe_device = next(model.parameters()).device
                probe_on_device = {k: v.to(probe_device) for k, v in probe_batch.items()}

                class AlignTrainer(Trainer):
                    def compute_loss(self, mdl, inputs, return_outputs=False, **kwargs):
                        outA = mdl(**inputs)
                        lossA = outA.loss
                        if lam == 0.0:
                            return (lossA, outA) if return_outputs else lossA
                        outB = mdl(**probe_on_device)
                        loss = lossA + lam * outB.loss
                        return (loss, outA) if return_outputs else loss

                args = TrainingArguments(
                    output_dir=os.path.join(output_dir, f"tmp_{model_name.split('/')[-1]}_{str(lam).replace('.', 'p')}_seed{seed}_s{steps}"),
                    per_device_train_batch_size=train_batch_size,
                    per_device_eval_batch_size=train_batch_size,
                    gradient_accumulation_steps=grad_accum,
                    learning_rate=learning_rate,
                    max_steps=steps,
                    logging_steps=logging_steps,
                    save_strategy="no",
                    report_to=[],
                    fp16=torch.cuda.is_available(),
                    remove_unused_columns=False,
                    dataloader_num_workers=0,
                    gradient_checkpointing=True,
                    seed=seed,
                    data_seed=seed,
                )

                trainer = AlignTrainer(model=model, args=args, train_dataset=trainA, data_collator=collator)
                t0 = time.time()
                trainer.train()
                elapsed = time.time() - t0

                lossA = float(trainer.evaluate(eval_dataset=evalA)["eval_loss"])
                lossB = float(trainer.evaluate(eval_dataset=evalB)["eval_loss"])
                gap = lossB - lossA
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())

                row = {
                    "model": model_name,
                    "lambda_align": float(lam),
                    "seed": int(seed),
                    "steps": int(steps),
                    "loss_A": lossA,
                    "loss_B": lossB,
                    "gap": gap,
                    "train_time_s": round(elapsed, 1),
                    "trainable_params": int(trainable),
                    "total_params": int(total),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                done.add(key)
                print(f"loss_A={lossA:.4f} | loss_B={lossB:.4f} | gap={gap:+.4f} | {elapsed:.0f}s")

                del trainer, model
                gpu_clean()

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(csv_path, index=False)
    return df



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-model regime-collapse robustness sweeps")
    parser.add_argument("--input-root", default=".", help="Directory containing regime_A.txt and regime_B S.txt")
    parser.add_argument("--output-dir", default="./results_large", help="Directory for outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to sweep")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--probe-bs", type=int, default=DEFAULT_PROBE_BS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--block-chars", type=int, default=DEFAULT_BLOCK_CHARS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token for gated models like Llama 2")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    maybe_set_hf_token_env(args.hf_token)
    ensure_dir(args.output_dir)
    data = prepare_language_data([args.input_root], block_chars=args.block_chars)

    summary = []
    for model_name in args.models:
        print(f"\nPreparing tokenizer and tokenized datasets for {model_name}")
        tokenizer = build_tokenizer(model_name, args.hf_token)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainA = TextTokenizedDataset(data.train_langA, tokenizer, args.max_length)
        trainB = TextTokenizedDataset(data.train_langB, tokenizer, args.max_length)
        evalA = TextTokenizedDataset(data.eval_langA, tokenizer, args.max_length)
        evalB = TextTokenizedDataset(data.eval_langB, tokenizer, args.max_length)

        csv_name = f"{model_name.split('/')[-1].replace('-', '_')}_LAMSWEEP.csv"
        df = run_large_model_sweep(
            model_name=model_name,
            tokenizer=tokenizer,
            collator=collator,
            trainA=trainA,
            trainB=trainB,
            evalA=evalA,
            evalB=evalB,
            output_dir=args.output_dir,
            csv_name=csv_name,
            lambdas=args.lambdas,
            seeds=args.seeds,
            steps_list=args.steps,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            grad_accum=args.grad_accum,
            probe_bs=args.probe_bs,
            logging_steps=args.logging_steps,
            hf_token=args.hf_token,
        )
        if len(df):
            best = df.sort_values(["gap", "loss_B"]).iloc[0].to_dict()
            summary.append({
                "model": model_name,
                "n_runs": int(len(df)),
                "best_lambda": float(best["lambda_align"]),
                "best_steps": int(best["steps"]),
                "best_gap": float(best["gap"]),
                "best_loss_B": float(best["loss_B"]),
            })

    summary_path = os.path.join(args.output_dir, "large_model_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
