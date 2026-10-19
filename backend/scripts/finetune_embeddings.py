"""Fine-tune the embedding model on pairs built from the train papers (app/finetune/pairs.py).

Contrastive training with in-batch negatives: each query should score its own
passage above the other passages in the batch. Same CLS pooling, normalisation
and query instruction as app/index/embeddings.py, so the result is a drop-in
replacement. CPU only; about 18 s per batch of 32 on the development machine.

Usage:
  python scripts/finetune_embeddings.py --out data/models/bge-small-ft-pilot --pairs 1000
Then build an index with it (scripts/build_index.py --model <out>) and run the eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from app.config import PAPERS_DIR, REPO_ROOT  # noqa: E402
from app.finetune.pairs import build_pairs, write_pairs  # noqa: E402
from app.index.embeddings import DEFAULT_MODEL, QUERY_INSTRUCTION  # noqa: E402


def embed(model, tokenizer, texts: list[str], max_tokens: int) -> torch.Tensor:
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
    return torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], dim=-1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8000, help="how many pairs to train on (from the shuffled pool)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.02)  # bge's own fine-tuning default
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--max-passage-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists; choose a new --out")

    split = json.loads((REPO_ROOT / "eval" / "split.json").read_text(encoding="utf-8"))
    pool = build_pairs(PAPERS_DIR, set(split["train"]), set(split["test"]), seed=args.seed)
    pairs = pool[: args.pairs]
    print(f"{len(pool)} candidate pairs from {len(split['train'])} train papers; using {len(pairs)} "
          f"({dict(Counter(p.kind for p in pairs))})")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL)
    model.train()
    steps_per_epoch = math.ceil(len(pairs) / args.batch)
    total = steps_per_epoch * args.epochs
    warmup = max(1, total // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))

    losses: list[float] = []
    started = time.monotonic()
    step = 0
    for epoch in range(args.epochs):
        order = pairs[:]
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            batch = order[start : start + args.batch]
            if len(batch) < 2:
                continue  # in-batch negatives need at least two pairs
            q = embed(model, tokenizer, [QUERY_INSTRUCTION + p.query for p in batch], args.max_query_tokens)
            d = embed(model, tokenizer, [p.passage for p in batch], args.max_passage_tokens)
            loss = torch.nn.functional.cross_entropy(q @ d.T / args.temperature, torch.arange(len(batch)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            losses.append(float(loss))
            step += 1
            if step % 5 == 0 or step == total:
                elapsed = time.monotonic() - started
                recent = sum(losses[-5:]) / len(losses[-5:])
                print(f"epoch {epoch + 1} step {step}/{total}  loss {recent:.3f}  "
                      f"elapsed {elapsed / 60:.1f} min  eta {elapsed / step * (total - step) / 60:.1f} min", flush=True)

    args.out.mkdir(parents=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    write_pairs(pairs, args.out / "pairs.jsonl")
    (args.out / "train.json").write_text(json.dumps({
        "base_model": DEFAULT_MODEL,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pairs": len(pairs), "pairs_by_kind": dict(Counter(p.kind for p in pairs)),
        "train_papers_sha256": hashlib.sha256("\n".join(sorted(split["train"])).encode()).hexdigest(),
        "steps": step, "minutes": round((time.monotonic() - started) / 60, 1),
        "loss_first_10": losses[:10], "loss_last_10": losses[-10:],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
