"""Run the full IrisSecure pipeline with content-addressed caching."""
import argparse
import sys
import time

from config import CFG
from src.caching import load_hash_index, load_manifest

from stages.s0_clean import run_clean
from stages.s1_preprocess import run_preprocess
from stages.s2_train import run_train
from stages.s3_embed import run_embed
from stages.s4_evaluate import run_evaluate


STAGES = [
    ("clean", run_clean, {}),
    ("preprocess", run_preprocess, {}),
    (
        "train",
        run_train,
        {"epochs": None, "batch_size": None, "grad_accum": None, "num_workers": None, "max_classes": None},
    ),
    ("embed", run_embed, {"batch_size": None, "num_workers": None}),
    ("evaluate", run_evaluate, {"split": "test", "max_impostor": None}),
]


def show_status():
    idx = load_hash_index()
    print(f"\n{'Stage':12s} {'Status':10s} {'Hash':18s} {'Duration':>10s}")
    print("-" * 60)
    for name, _, _ in STAGES:
        m = load_manifest(name)
        if m and idx.get(name) == m.get("input_hash"):
            print(f"{name:12s} {'cached':10s} {m['input_hash']:18s} {m.get('duration', 0):>9.1f}s")
        elif m:
            print(f"{name:12s} {'stale':10s} {m['input_hash']:18s} {'':>10s}")
        else:
            print(f"{name:12s} {'never':10s} {'-':18s} {'':>10s}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_stage", type=str, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_classes", type=int, default=None)
    args = ap.parse_args()

    CFG.ensure_dirs()

    if args.status:
        show_status()
        return

    names = [n for n, _, _ in STAGES]
    if args.only:
        if args.only not in names:
            sys.exit(f"Unknown stage '{args.only}'. Choose from {names}")
        to_run = [args.only]
    elif args.from_stage:
        if args.from_stage not in names:
            sys.exit(f"Unknown stage '{args.from_stage}'. Choose from {names}")
        to_run = names[names.index(args.from_stage):]
    else:
        to_run = names

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.max_classes is not None:
        overrides["max_classes"] = args.max_classes

    t0 = time.time()
    results = []
    for name, fn, defaults in STAGES:
        if name not in to_run:
            continue
        kwargs = {k: v for k, v in (defaults or {}).items() if v is not None}
        kwargs.update({k: v for k, v in overrides.items() if k in (defaults or {})})
        res = fn(force=args.force, **kwargs)
        results.append(res)
        print(res)

    print(f"\n{'='*66}")
    print(f"Pipeline finished in {time.time() - t0:.1f}s")
    print(f"{'='*66}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
