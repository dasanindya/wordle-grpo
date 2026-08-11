"""Dataset loading and word-list caching for Wordle-GRPO.

The reward functions call ``pd.read_csv(example["word_list"])`` on every scored
completion. We download/copy that dictionary once to a local CSV and rewrite the
column to point at it, so scoring is fast and works offline during training.
"""
import os
import shutil
import urllib.request

import pandas as pd
from datasets import load_dataset

import config


def cache_word_list(dataset, path: str = config.WORD_LIST_PATH) -> str:
    """Ensure a local copy of the valid-word dictionary exists; return its path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wl_value = str(dataset[0]["word_list"])

    if os.path.exists(path):
        pass  # already cached
    elif wl_value.startswith(("http://", "https://")):
        print(f"[data] downloading word list from {wl_value}")
        urllib.request.urlretrieve(wl_value, path)
    elif os.path.exists(wl_value):
        shutil.copy(wl_value, path)
    else:
        # Column may already be a usable local path.
        path = wl_value

    wl = pd.read_csv(path)
    assert "Word" in wl.columns, f"expected a 'Word' column, got {list(wl.columns)}"
    print(f"[data] word list: {len(wl)} words at {path}")
    return path


def load_grpo_dataset(subset: bool = True):
    """Load the GRPO dataset, cache the word list, and remap the column.

    If ``subset`` and ``config.N_TRAIN`` is set, returns a shuffled subset (used by
    QUICK mode).
    """
    ds = load_dataset(config.GRPO_DATASET_ID, split="train")
    local = cache_word_list(ds, config.WORD_LIST_PATH)
    ds = ds.map(lambda ex: {"word_list": local})

    if subset and config.N_TRAIN is not None:
        n = min(config.N_TRAIN, len(ds))
        ds = ds.shuffle(seed=42).select(range(n))
        print(f"[data] using subset of {n} rows")
    else:
        print(f"[data] using full dataset: {len(ds)} rows")
    return ds


def load_sft_dataset(subset: bool = True):
    """Load the Wordle SFT dataset (Part 3)."""
    ds = load_dataset(config.SFT_DATASET_ID, split="train")
    if subset and config.N_TRAIN is not None:
        n = min(config.N_TRAIN, len(ds))
        ds = ds.shuffle(seed=42).select(range(n))
        print(f"[data] SFT subset of {n} rows")
    return ds


if __name__ == "__main__":
    ds = load_grpo_dataset()
    print("columns:", ds.column_names)
