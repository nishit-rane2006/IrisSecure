import argparse
from pathlib import Path

import numpy as np

from config import CFG
from register import embed_image, load_gallery, load_model
from src.security import decrypt_embedding, derive_key
from src.utils import get_device


def _materialize(rec, password):
    emb = rec["embedding"]
    if isinstance(emb, dict):
        if not password:
            raise RuntimeError("Identity is encrypted; supply --password")
        emb = decrypt_embedding(emb, derive_key(password, bytes.fromhex(rec["salt"])))
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--image1", type=Path)
    ap.add_argument("--image2", type=Path)
    ap.add_argument("--against", type=str)
    ap.add_argument("--identify", action="store_true")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=CFG.verify_threshold)
    ap.add_argument("--password", type=str, default=None)
    args = ap.parse_args()

    device = get_device()
    model = load_model(device)

    if args.image1 and args.image2:
        e1 = embed_image(model, device, args.image1)
        e2 = embed_image(model, device, args.image2)
        sim = float(np.dot(e1, e2))
        print(f"Pairwise cosine similarity: {sim:.4f}  ({'MATCH' if sim >= args.threshold else 'NO MATCH'})")
        return

    if not args.image:
        ap.error("--image is required (or use --image1/--image2)")

    probe = embed_image(model, device, args.image)

    if args.against:
        gallery = load_gallery()
        if args.against not in gallery["users"]:
            print(f"Unknown identity: {args.against}")
            return
        ref = _materialize(gallery["users"][args.against], args.password)
        sim = float(np.dot(probe, ref))
        ok = sim >= args.threshold
        print(f"1:1 verification against '{args.against}': sim={sim:.4f}  threshold={args.threshold}  →  {'ACCEPT' if ok else 'REJECT'}")
        return

    if args.identify:
        gallery = load_gallery()
        scores = []
        for uid, rec in gallery["users"].items():
            try:
                ref = _materialize(rec, args.password)
            except RuntimeError:
                continue
            scores.append((uid, float(np.dot(probe, ref))))
        scores.sort(key=lambda x: -x[1])
        print(f"Top-{args.top_k} matches (threshold={args.threshold}):")
        for uid, s in scores[:args.top_k]:
            mark = "✓" if s >= args.threshold else " "
            print(f"  [{mark}] {uid:20s}  sim={s:.4f}")
        return

    ap.error("Specify --against, --identify, or --image1/--image2")


if __name__ == "__main__":
    main()
