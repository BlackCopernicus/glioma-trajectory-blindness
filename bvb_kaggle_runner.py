#!/usr/bin/env python3
"""
BvB / Regime Collapse - Kaggle reproduction runner

This script consolidates the code spine recovered from:
1) the Kaggle v4 notebook (core GPT-2 small experiments, figures, diagnostics),
2) the Colab lambda-sweep notebook v2 (missing GPT-2 small lambda sweep), and
3) the notebook cells for GPT-2 medium lambda sweep + calibration runs.

Design goals:
- one runnable .py file for GitHub
- faithful to the recovered code structure and hyperparameters
- resumable: CSVs autosave and completed runs are skipped
- modular: run the whole pipeline or selected parts from the CLI

Expected data files (somewhere under --input-root, default: current dir):
- regime_A.txt
- regime_B S.txt
- math_regime_A_CME.tex
- math_regime_B_PCM.tex

Primary outputs are written to --output-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)


# -----------------------------------------------------------------------------
# Defaults recovered from the uploaded artifacts
# -----------------------------------------------------------------------------
MODEL_SMALL = "gpt2"
MODEL_MEDIUM = "gpt2-medium"
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_STEPS = [150, 300, 450, 600]
DEFAULT_LR = 5e-5
DEFAULT_BATCH = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_PROBE_BS = 8
DEFAULT_BLOCK_CHARS = 1500
DEFAULT_LAMBDA_FIX = 0.05
DEFAULT_SMALL_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
DEFAULT_SMALL_SWEEP_CONFIGS = [(600, [0, 1, 2]), (300, [0])]

# Medium settings recovered from notebook cells / calibration runs
DEFAULT_MEDIUM_BATCH = 1
DEFAULT_MEDIUM_GRAD_ACCUM = 16
DEFAULT_MEDIUM_EVAL_BS = 1
DEFAULT_MEDIUM_LOGGING_STEPS = 50
DEFAULT_MEDIUM_SWEEP_LAMBDAS = [0.0, 0.0025, 0.0050, 0.0500]
DEFAULT_MEDIUM_SWEEP_STEPS = 300
DEFAULT_MEDIUM_SWEEP_SEED = 0
DEFAULT_MEDIUM_CALIBRATION_LAMBDA = 0.0025
DEFAULT_MEDIUM_CALIBRATION_SEEDS = [1, 2]
DEFAULT_MEDIUM_CALIBRATION_STEPS = 300


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
    """Recovered from Kaggle v4 notebook."""
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


def json_dump(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class PreparedData:
    tok_small: AutoTokenizer
    collator_small: DataCollatorForLanguageModeling
    train_langA: Dataset
    eval_langA: Dataset
    train_langB: Dataset
    eval_langB: Dataset
    train_mathA: Dataset
    eval_mathA: Dataset
    train_mathB: Dataset
    eval_mathB: Dataset


# -----------------------------------------------------------------------------
# Data prep
# -----------------------------------------------------------------------------

def prepare_datasets(input_roots: Sequence[str], block_chars: int = DEFAULT_BLOCK_CHARS) -> PreparedData:
    lang_a_path = find_file("regime_A.txt", input_roots)
    lang_b_path = find_file("regime_B S.txt", input_roots)
    math_a_path = find_file("math_regime_A_CME.tex", input_roots)
    math_b_path = find_file("math_regime_B_PCM.tex", input_roots)

    lang_A = load_text(lang_a_path)
    lang_B = load_text(lang_b_path)
    math_A = clean_latex(load_text(math_a_path))
    math_B = clean_latex(load_text(math_b_path))

    tok = AutoTokenizer.from_pretrained(MODEL_SMALL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    def tok_blocks(ds: Dataset, label: str = "") -> Dataset:
        def f(batch):
            out = tok(batch["text"], truncation=True, max_length=1024)
            return {"input_ids": out["input_ids"]}

        ds2 = ds.map(f, batched=True, remove_columns=["text"])
        ds2 = ds2.filter(lambda x: len(x["input_ids"]) > 8)
        print(f"{label}: {len(ds2)} tokenized blocks")
        return ds2

    print("Tokenizing datasets...")
    langA = tok_blocks(make_blocks(lang_A, block_chars), "langA")
    langB = tok_blocks(make_blocks(lang_B, block_chars), "langB")
    mathA = tok_blocks(make_blocks(math_A, block_chars), "mathA")
    mathB = tok_blocks(make_blocks(math_B, block_chars), "mathB")

    train_langA, eval_langA = split_ds(langA, n_train=600, n_eval=256, seed=0)
    train_langB, eval_langB = split_ds(langB, n_train=40, n_eval=39, seed=0)

    n_mathA_total = len(mathA)
    n_mathB_total = len(mathB)
    n_mathA_train = min(200, int(n_mathA_total * 0.65))
    n_mathA_eval = min(100, n_mathA_total - n_mathA_train)
    n_mathB_train = min(40, int(n_mathB_total * 0.5))
    n_mathB_eval = min(n_mathB_total - n_mathB_train, 40)

    train_mathA, eval_mathA = split_ds(mathA, n_train=n_mathA_train, n_eval=n_mathA_eval, seed=0)
    train_mathB, eval_mathB = split_ds(mathB, n_train=n_mathB_train, n_eval=n_mathB_eval, seed=0)

    print("Prepared splits:")
    print(f"LANG A train/eval = {len(train_langA)}/{len(eval_langA)}")
    print(f"LANG B train/eval = {len(train_langB)}/{len(eval_langB)}")
    print(f"MATH A train/eval = {len(train_mathA)}/{len(eval_mathA)}")
    print(f"MATH B train/eval = {len(train_mathB)}/{len(eval_mathB)}")

    return PreparedData(
        tok_small=tok,
        collator_small=collator,
        train_langA=train_langA,
        eval_langA=eval_langA,
        train_langB=train_langB,
        eval_langB=eval_langB,
        train_mathA=train_mathA,
        eval_mathA=eval_mathA,
        train_mathB=train_mathB,
        eval_mathB=eval_mathB,
    )


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------

def _load_existing(csv_path: str) -> List[dict]:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).to_dict("records")
    return []


def _done_keyed(rows: Iterable[dict], key_cols: Sequence[str]) -> set:
    done = set()
    for r in rows:
        done.add(tuple(float(r[c]) if c == "lambda_align" else int(r[c]) if c in {"seed", "steps"} else r[c] for c in key_cols))
    return done


def run_experiment(
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    collator: DataCollatorForLanguageModeling,
    trainA: Dataset,
    trainB: Dataset,
    evalA: Dataset,
    evalB: Dataset,
    tag: str,
    output_dir: str,
    csv_name: str,
    lambda_align: float = 0.0,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps_list: Sequence[int] = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LR,
    train_batch_size: int = DEFAULT_BATCH,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    probe_bs: int = DEFAULT_PROBE_BS,
    eval_batch_size: Optional[int] = None,
    logging_steps: int = 50,
    gradient_checkpointing: bool = False,
    autosave: bool = True,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = os.path.join(output_dir, csv_name)
    rows = _load_existing(csv_path)
    done = _done_keyed(rows, ["tag", "lambda_align", "seed", "steps"]) if rows else set()

    for seed in seeds:
        probe_ds = trainB.shuffle(seed=seed).select(range(min(probe_bs, len(trainB))))
        probe_batch = collator([probe_ds[i] for i in range(len(probe_ds))])

        for steps in steps_list:
            key = (tag, float(lambda_align), int(seed), int(steps))
            if key in done:
                print(f"[skip] {key} already present in {csv_name}")
                continue

            print("=" * 72)
            print(f"{tag} | model={model_name} | λ={lambda_align:.4f} | seed={seed} | steps={steps}")
            print("=" * 72)

            set_all_seeds(seed)
            gpu_clean()

            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            if gradient_checkpointing:
                model.config.use_cache = False
                model.gradient_checkpointing_enable()

            probe_on_device = {k: v.to(device) for k, v in probe_batch.items()}
            lam = float(lambda_align)

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
                output_dir=os.path.join(output_dir, f"tmp_{tag}_{str(lambda_align).replace('.', 'p')}_seed{seed}_s{steps}"),
                per_device_train_batch_size=train_batch_size,
                gradient_accumulation_steps=grad_accum,
                per_device_eval_batch_size=eval_batch_size or train_batch_size,
                learning_rate=learning_rate,
                max_steps=steps,
                logging_steps=logging_steps,
                save_strategy="no",
                report_to=[],
                fp16=torch.cuda.is_available(),
                remove_unused_columns=False,
                dataloader_num_workers=0,
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

            row = {
                "tag": tag,
                "model": model_name,
                "lambda_align": float(lambda_align),
                "seed": int(seed),
                "steps": int(steps),
                "loss_A": lossA,
                "loss_B": lossB,
                "gap": gap,
                "train_time_s": round(elapsed, 1),
            }
            rows.append(row)
            done.add(key)
            if autosave:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"loss_A={lossA:.4f} | loss_B={lossB:.4f} | gap={gap:+.4f} | {elapsed:.0f}s")

            del trainer, model
            gpu_clean()

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(csv_path, index=False)
    return df


# -----------------------------------------------------------------------------
# Predictability diagnostic / figures / summaries
# -----------------------------------------------------------------------------

def eval_loss_quick(model, dataset: Dataset, collator, device: str, batch_size: int = 8) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            idx = range(i, min(i + batch_size, len(dataset)))
            batch = collator([dataset[j] for j in idx])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
    return float(np.mean(losses))


def run_predictability_diagnostic(data: PreparedData, output_dir: str) -> dict:
    ensure_dir(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_SMALL)
    base_model.resize_token_embeddings(len(data.tok_small))
    base_model.to(device)
    base_model.eval()

    base_lA = eval_loss_quick(base_model, data.eval_langA, data.collator_small, device, batch_size=8)
    base_lB = eval_loss_quick(base_model, data.eval_langB, data.collator_small, device, batch_size=8)
    base_mA = eval_loss_quick(base_model, data.eval_mathA, data.collator_small, device, batch_size=8)
    base_mB = eval_loss_quick(base_model, data.eval_mathB, data.collator_small, device, batch_size=8)

    delta_lang = abs(base_lB - base_lA)
    delta_math = abs(base_mB - base_mA)
    out = {
        "lang_A": base_lA,
        "lang_B": base_lB,
        "math_A": base_mA,
        "math_B": base_mB,
        "delta_lang": delta_lang,
        "delta_math": delta_math,
        "predicted_worse": "Language" if delta_lang > delta_math else "Mathematics",
    }
    json_dump(out, os.path.join(output_dir, "Coupling_Strength_Predicts_Gap_Magnitude.json"))
    del base_model
    gpu_clean()
    return out


def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures_and_summary(output_dir: str, include_medium: bool = True) -> None:
    ensure_dir(output_dir)
    plt = _safe_import_matplotlib()

    def load_optional(name: str) -> Optional[pd.DataFrame]:
        path = os.path.join(output_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    df_lang_base = load_optional("LANG_BASE_lam0p0_log.csv")
    df_math_base = load_optional("MATH_BASE_lam0p0_log.csv")
    df_lang_fix = load_optional("LANG_FIX_lam0p15_log.csv")
    df_lang_reverse = load_optional("LANG_REVERSE_lam0p0_log.csv")
    df_sweep = load_optional("LANG_SWEEP_log.csv")
    df_med = load_optional("GPT2_MEDIUM_LAMSWEEP.csv") or load_optional("GPT2 MEDIUM LAMSWEEP recovered.csv")
    cal = load_optional("CALIBRATION_MEDIUM_RUNS.csv")

    dfs = []
    for df in [df_lang_base, df_math_base, df_lang_fix, df_lang_reverse, df_sweep, df_med, cal]:
        if df is not None:
            dfs.append(df)
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(os.path.join(output_dir, "ALL_RESULTS.csv"), index=False)

    if df_lang_base is None or df_math_base is None or df_lang_fix is None:
        print("Skipping figure generation: base/fix CSVs not all present yet.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    })

    # Figure 1: scissor baseline vs fix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    agg = df_lang_base.groupby("steps").agg(
        mean_A=("loss_A", "mean"), std_A=("loss_A", "std"),
        mean_B=("loss_B", "mean"), std_B=("loss_B", "std"),
    ).reset_index()
    ax = axes[0]
    ax.plot(agg["steps"], agg["mean_A"], "o-", lw=2, ms=6, label="Train regime (A)")
    ax.fill_between(agg["steps"], agg["mean_A"] - agg["std_A"], agg["mean_A"] + agg["std_A"], alpha=0.15)
    ax.plot(agg["steps"], agg["mean_B"], "s-", lw=2, ms=6, label="Deploy regime (B)")
    ax.fill_between(agg["steps"], agg["mean_B"] - agg["std_B"], agg["mean_B"] + agg["std_B"], alpha=0.15)
    ax.set_xlabel("Fine-tuning steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("(a) Baseline: no stabilization", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)

    agg_f = df_lang_fix.groupby("steps").agg(
        mean_A=("loss_A", "mean"), std_A=("loss_A", "std"),
        mean_B=("loss_B", "mean"), std_B=("loss_B", "std"),
    ).reset_index()
    ax = axes[1]
    ax.plot(agg_f["steps"], agg_f["mean_A"], "o-", lw=2, ms=6, label="Train regime (A)")
    ax.fill_between(agg_f["steps"], agg_f["mean_A"] - agg_f["std_A"], agg_f["mean_A"] + agg_f["std_A"], alpha=0.15)
    ax.plot(agg_f["steps"], agg_f["mean_B"], "s-", lw=2, ms=6, label="Deploy regime (B)")
    ax.fill_between(agg_f["steps"], agg_f["mean_B"] - agg_f["std_B"], agg_f["mean_B"] + agg_f["std_B"], alpha=0.15)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("Fine-tuning steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("(b) With stabilization: λ = 0.15", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_scissor.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "fig1_scissor.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: language vs math gap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    agg_lg = df_lang_base.groupby("steps").agg(mean_gap=("gap", "mean"), std_gap=("gap", "std")).reset_index()
    agg_mg = df_math_base.groupby("steps").agg(mean_gap=("gap", "mean"), std_gap=("gap", "std")).reset_index()
    ax = axes[0]
    ax.plot(agg_lg["steps"], agg_lg["mean_gap"], "o-", lw=2.5, ms=7)
    ax.fill_between(agg_lg["steps"], agg_lg["mean_gap"] - agg_lg["std_gap"], agg_lg["mean_gap"] + agg_lg["std_gap"], alpha=0.2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Gap = Loss_B - Loss_A")
    ax.set_title("(a) Language: high coupling strength", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.axhline(y=0, color="k", ls="--", alpha=0.3)
    ax = axes[1]
    ax.plot(agg_mg["steps"], agg_mg["mean_gap"], "o-", lw=2.5, ms=7)
    ax.fill_between(agg_mg["steps"], agg_mg["mean_gap"] - agg_mg["std_gap"], agg_mg["mean_gap"] + agg_mg["std_gap"], alpha=0.2)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("Steps")
    ax.set_ylabel("Gap = Loss_B - Loss_A")
    ax.set_title("(b) Mathematics: low coupling strength", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.axhline(y=0, color="k", ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_coupling.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "fig2_coupling.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 3: overlay
    fig, ax = plt.subplots(figsize=(8, 5.5))
    agg_fg = df_lang_fix.groupby("steps").agg(mean_gap=("gap", "mean"), std_gap=("gap", "std")).reset_index()
    ax.plot(agg_lg["steps"], agg_lg["mean_gap"], "o-", lw=2.5, ms=7, label="Language, λ=0")
    ax.fill_between(agg_lg["steps"], agg_lg["mean_gap"] - agg_lg["std_gap"], agg_lg["mean_gap"] + agg_lg["std_gap"], alpha=0.15)
    ax.plot(agg_fg["steps"], agg_fg["mean_gap"], "s--", lw=2.5, ms=7, label="Language, λ=0.15")
    ax.fill_between(agg_fg["steps"], agg_fg["mean_gap"] - agg_fg["std_gap"], agg_fg["mean_gap"] + agg_fg["std_gap"], alpha=0.15)
    ax.plot(agg_mg["steps"], agg_mg["mean_gap"], "^-", lw=2.5, ms=7, label="Mathematics, λ=0")
    ax.fill_between(agg_mg["steps"], agg_mg["mean_gap"] - agg_mg["std_gap"], agg_mg["mean_gap"] + agg_mg["std_gap"], alpha=0.15)
    ax.axhline(y=0, color="k", ls="--", alpha=0.3, lw=0.8)
    ax.set_xlabel("Fine-tuning steps")
    ax.set_ylabel("Regime gap = Loss_B - Loss_A")
    ax.set_title("Regime Collapse: Prediction and Mitigation", fontweight="bold")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_all_gaps.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(output_dir, "fig3_all_gaps.pdf"), bbox_inches="tight")
    plt.close(fig)

    if df_sweep is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sw = df_sweep.groupby("lambda_align").agg(
            mean_gap=("gap", "mean"), std_gap=("gap", "std"),
            mean_A=("loss_A", "mean"), std_A=("loss_A", "std"),
            mean_B=("loss_B", "mean"), std_B=("loss_B", "std"),
        ).reset_index()
        ax = axes[0]
        ax.errorbar(sw["lambda_align"], sw["mean_A"], yerr=sw["std_A"], fmt="o-", lw=2, ms=6, capsize=4, label="Loss A")
        ax.errorbar(sw["lambda_align"], sw["mean_B"], yerr=sw["std_B"], fmt="s-", lw=2, ms=6, capsize=4, label="Loss B")
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Loss")
        ax.set_title("(a) Losses vs lambda", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.2)
        ax = axes[1]
        ax.errorbar(sw["lambda_align"], sw["mean_gap"], yerr=sw["std_gap"], fmt="D-", lw=2.5, ms=7, capsize=4)
        ax.axhline(y=0, color="k", ls="--", alpha=0.3)
        ax.axvline(x=0.15, ls=":", alpha=0.6, label="λ*=0.15")
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Gap")
        ax.set_title("(b) Gap vs lambda", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_sweep.png"), bbox_inches="tight", dpi=300)
        fig.savefig(os.path.join(output_dir, "fig4_sweep.pdf"), bbox_inches="tight")
        plt.close(fig)

    if include_medium and df_med is not None:
        med_summary = df_med.groupby("lambda_align", as_index=False).agg(
            mean_loss_A=("loss_A", "mean"),
            mean_loss_B=("loss_B", "mean"),
            mean_gap=("gap", "mean"),
            n=("seed", "count"),
        )
        med_summary.to_csv(os.path.join(output_dir, "GPT2_MEDIUM_LAMSWEEP_summary.csv"), index=False)


def medium_calibration_summary(output_dir: str, csv_name: str = "GPT2_MEDIUM_LAMSWEEP.csv", step: int = 300, max_lam_for_fit: float = 0.005) -> pd.DataFrame:
    path = os.path.join(output_dir, csv_name)
    if not os.path.exists(path):
        alt = os.path.join(output_dir, "GPT2 MEDIUM LAMSWEEP recovered.csv")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Could not find {csv_name} or recovered medium sweep CSV in {output_dir}")

    df = pd.read_csv(path)
    df = df[df["steps"] == step].copy()
    df_small = (
        df[df["lambda_align"] <= max_lam_for_fit]
        .groupby("lambda_align", as_index=False)["gap"]
        .mean()
        .sort_values("lambda_align")
    )
    ser0 = df_small.loc[df_small["lambda_align"] == 0.0, "gap"]
    if len(ser0) == 0:
        raise ValueError("No lambda_align == 0.0 row found in medium sweep data.")
    baseline_gap = float(ser0.iloc[0])
    coeffs = np.polyfit(df_small["lambda_align"].values, df_small["gap"].values, 2)
    a, b, c = coeffs
    lambda_star = float(-b / (2 * a))
    best_idx = df_small["gap"].abs().idxmin()
    min_gap = float(df_small.loc[best_idx, "gap"])
    reduction_pct = 100.0 * (1.0 - abs(min_gap) / abs(baseline_gap)) if abs(baseline_gap) > 1e-12 else np.nan
    out = pd.DataFrame([{
        "model": MODEL_MEDIUM,
        "steps": step,
        "baseline_gap": baseline_gap,
        "min_gap_observed": min_gap,
        "reduction_percent": reduction_pct,
        "lambda_star_estimated": lambda_star,
        "quadratic_a": float(a),
        "quadratic_b": float(b),
        "quadratic_c": float(c),
        "curvature_2a": float(2 * a),
    }])
    out.to_csv(os.path.join(output_dir, "CALIBRATION_SUMMARY.csv"), index=False)
    return out


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run_small_core(data: PreparedData, output_dir: str) -> None:
    run_experiment(
        model_name=MODEL_SMALL,
        tokenizer=data.tok_small,
        collator=data.collator_small,
        trainA=data.train_langA,
        trainB=data.train_langB,
        evalA=data.eval_langA,
        evalB=data.eval_langB,
        tag="LANG_BASE",
        output_dir=output_dir,
        csv_name="LANG_BASE_lam0p0_log.csv",
        lambda_align=0.0,
    )
    run_experiment(
        model_name=MODEL_SMALL,
        tokenizer=data.tok_small,
        collator=data.collator_small,
        trainA=data.train_mathA,
        trainB=data.train_mathB,
        evalA=data.eval_mathA,
        evalB=data.eval_mathB,
        tag="MATH_BASE",
        output_dir=output_dir,
        csv_name="MATH_BASE_lam0p0_log.csv",
        lambda_align=0.0,
    )
    run_experiment(
        model_name=MODEL_SMALL,
        tokenizer=data.tok_small,
        collator=data.collator_small,
        trainA=data.train_langA,
        trainB=data.train_langB,
        evalA=data.eval_langA,
        evalB=data.eval_langB,
        tag="LANG_FIX",
        output_dir=output_dir,
        csv_name="LANG_FIX_lam0p15_log.csv",
        lambda_align=DEFAULT_LAMBDA_FIX,
    )
    run_experiment(
        model_name=MODEL_SMALL,
        tokenizer=data.tok_small,
        collator=data.collator_small,
        trainA=data.train_langB,
        trainB=data.train_langA,
        evalA=data.eval_langB,
        evalB=data.eval_langA,
        tag="LANG_REVERSE",
        output_dir=output_dir,
        csv_name="LANG_REVERSE_lam0p0_log.csv",
        lambda_align=0.0,
        seeds=[0],
        steps_list=[600],
    )


def run_small_sweep(data: PreparedData, output_dir: str, lambdas: Sequence[float] = DEFAULT_SMALL_SWEEP) -> pd.DataFrame:
    rows = []
    csv_path = os.path.join(output_dir, "LANG_SWEEP_log.csv")
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
    done = _done_keyed(rows, ["tag", "lambda_align", "seed", "steps"]) if rows else set()

    for sweep_steps, sweep_seeds in DEFAULT_SMALL_SWEEP_CONFIGS:
        for lam in lambdas:
            for seed in sweep_seeds:
                key = ("LANG_SWEEP", float(lam), int(seed), int(sweep_steps))
                if key in done:
                    print(f"[skip] {key} already present in LANG_SWEEP_log.csv")
                    continue
                df = run_experiment(
                    model_name=MODEL_SMALL,
                    tokenizer=data.tok_small,
                    collator=data.collator_small,
                    trainA=data.train_langA,
                    trainB=data.train_langB,
                    evalA=data.eval_langA,
                    evalB=data.eval_langB,
                    tag="LANG_SWEEP",
                    output_dir=output_dir,
                    csv_name="LANG_SWEEP_log.csv",
                    lambda_align=float(lam),
                    seeds=[seed],
                    steps_list=[int(sweep_steps)],
                )
                rows = df.to_dict("records")
                done = _done_keyed(rows, ["tag", "lambda_align", "seed", "steps"])
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(csv_path, index=False)
    return df


def run_medium_pipeline(data: PreparedData, output_dir: str, medium_lambdas: Sequence[float] = DEFAULT_MEDIUM_SWEEP_LAMBDAS) -> None:
    tok_med = AutoTokenizer.from_pretrained(MODEL_MEDIUM, use_fast=True)
    if tok_med.pad_token is None:
        tok_med.pad_token = tok_med.eos_token
    collator_med = DataCollatorForLanguageModeling(tok_med, mlm=False)

    run_experiment(
        model_name=MODEL_MEDIUM,
        tokenizer=tok_med,
        collator=collator_med,
        trainA=data.train_langA,
        trainB=data.train_langB,
        evalA=data.eval_langA,
        evalB=data.eval_langB,
        tag="MED_LANG",
        output_dir=output_dir,
        csv_name="GPT2_MEDIUM_LAMSWEEP.csv",
        lambda_align=medium_lambdas[0],
        seeds=[DEFAULT_MEDIUM_SWEEP_SEED],
        steps_list=[DEFAULT_MEDIUM_SWEEP_STEPS],
        train_batch_size=DEFAULT_MEDIUM_BATCH,
        grad_accum=DEFAULT_MEDIUM_GRAD_ACCUM,
        eval_batch_size=DEFAULT_MEDIUM_EVAL_BS,
        logging_steps=DEFAULT_MEDIUM_LOGGING_STEPS,
        gradient_checkpointing=True,
        probe_bs=DEFAULT_PROBE_BS,
    )
    for lam in medium_lambdas[1:]:
        run_experiment(
            model_name=MODEL_MEDIUM,
            tokenizer=tok_med,
            collator=collator_med,
            trainA=data.train_langA,
            trainB=data.train_langB,
            evalA=data.eval_langA,
            evalB=data.eval_langB,
            tag="MED_LANG",
            output_dir=output_dir,
            csv_name="GPT2_MEDIUM_LAMSWEEP.csv",
            lambda_align=float(lam),
            seeds=[DEFAULT_MEDIUM_SWEEP_SEED],
            steps_list=[DEFAULT_MEDIUM_SWEEP_STEPS],
            train_batch_size=DEFAULT_MEDIUM_BATCH,
            grad_accum=DEFAULT_MEDIUM_GRAD_ACCUM,
            eval_batch_size=DEFAULT_MEDIUM_EVAL_BS,
            logging_steps=DEFAULT_MEDIUM_LOGGING_STEPS,
            gradient_checkpointing=True,
            probe_bs=DEFAULT_PROBE_BS,
        )

    # calibration runs at λ = 0.0025 for seeds 1,2
    cal_csv = os.path.join(output_dir, "CALIBRATION_MEDIUM_RUNS.csv")
    cal_rows = _load_existing(cal_csv)
    cal_done = _done_keyed([
        {**r, "tag": "CAL_MED"} for r in cal_rows
    ], ["tag", "lambda_align", "seed", "steps"]) if cal_rows else set()

    for seed in DEFAULT_MEDIUM_CALIBRATION_SEEDS:
        key = ("CAL_MED", float(DEFAULT_MEDIUM_CALIBRATION_LAMBDA), int(seed), int(DEFAULT_MEDIUM_CALIBRATION_STEPS))
        if key in cal_done:
            print(f"[skip] calibration {key} already present")
            continue
        df = run_experiment(
            model_name=MODEL_MEDIUM,
            tokenizer=tok_med,
            collator=collator_med,
            trainA=data.train_langA,
            trainB=data.train_langB,
            evalA=data.eval_langA,
            evalB=data.eval_langB,
            tag="CAL_MED",
            output_dir=output_dir,
            csv_name="CALIBRATION_MEDIUM_RUNS.csv",
            lambda_align=DEFAULT_MEDIUM_CALIBRATION_LAMBDA,
            seeds=[seed],
            steps_list=[DEFAULT_MEDIUM_CALIBRATION_STEPS],
            train_batch_size=DEFAULT_MEDIUM_BATCH,
            grad_accum=DEFAULT_MEDIUM_GRAD_ACCUM,
            eval_batch_size=DEFAULT_MEDIUM_EVAL_BS,
            logging_steps=DEFAULT_MEDIUM_LOGGING_STEPS,
            gradient_checkpointing=True,
            probe_bs=4,
        )
        # Export per-seed csvs matching notebook convention
        match = df[(df["seed"] == seed) & (df["steps"] == DEFAULT_MEDIUM_CALIBRATION_STEPS) & (df["lambda_align"] == DEFAULT_MEDIUM_CALIBRATION_LAMBDA)]
        if len(match):
            match.drop(columns=[c for c in ["tag", "model"] if c in match.columns], inplace=False).to_csv(
                os.path.join(output_dir, f"fb_medium_lr0.0025_seed{seed}_steps300.csv"), index=False
            )

    medium_calibration_summary(output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full BvB / regime-collapse reproduction runner")
    parser.add_argument("--input-root", action="append", default=[], help="Root directory to search for input text/tex files. Can be repeated.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for CSVs, figures, and summaries")
    parser.add_argument(
        "--mode",
        choices=["all", "small-core", "small-sweep", "medium", "figures", "diagnostic"],
        default="all",
        help="Which part of the pipeline to run",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.output_dir)
    input_roots = list(dict.fromkeys(args.input_root + [os.getcwd(), "/kaggle/input", "/content", "/mnt/data"]))

    if args.mode in {"all", "small-core", "small-sweep", "medium", "diagnostic"}:
        data = prepare_datasets(input_roots=input_roots)
    else:
        data = None

    if args.mode in {"all", "small-core"}:
        run_small_core(data, args.output_dir)

    if args.mode in {"all", "small-sweep"}:
        run_small_sweep(data, args.output_dir)

    if args.mode in {"all", "medium"}:
        run_medium_pipeline(data, args.output_dir)

    if args.mode in {"all", "diagnostic"}:
        run_predictability_diagnostic(data, args.output_dir)

    if args.mode in {"all", "figures"}:
        make_figures_and_summary(args.output_dir, include_medium=True)

    print(f"Done. Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
