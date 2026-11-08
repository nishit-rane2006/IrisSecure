import argparse
import pickle
import secrets
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from src.dataset import build_eval_transform, preprocess_iris
from src.model import IrisEmbeddingNet
from src.security import apply_projection, decrypt_embedding, derive_key, encrypt_embedding, random_orthogonal
from src.utils import get_device

GALLERY_PATH = CFG.db_dir / "gallery.pkl"


def load_model(device):
    ckpt_path = CFG.work_dir / "artifacts" / "train" / "best_model.pth"
    if not ckpt_path.exists():
        raise SystemExit("No trained model found. Run `python pipeline.py` first.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = IrisEmbeddingNet(ckpt["num_classes"], ckpt["embedding_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]), strict=False)
    if CFG.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def load_gallery():
    if GALLERY_PATH.exists():
        with open(GALLERY_PATH, "rb") as f:
            return pickle.load(f)
    return {"version": 1, "users": {}}


def save_gallery(g):
    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GALLERY_PATH, "wb") as f:
        pickle.dump(g, f)


@torch.no_grad()
def embed_image(model, device, image_path: Path, tta: bool = True) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    img = preprocess_iris(img, CFG.image_size)
    tf = build_eval_transform(CFG.image_size)
    t = tf(image=img)["image"].unsqueeze(0).to(device)
    e = model.embed(t)
    if tta:
        e = F.normalize(e + model.embed(torch.flip(t, dims=[3])), dim=1)
    return e.squeeze(0).cpu().numpy().astype(np.float32)


def average_embeddings(model, device, image_dir: Path) -> np.ndarray:
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not paths:
        raise FileNotFoundError(f"No images in {image_dir}")
    embs = np.stack([embed_image(model, device, p) for p in paths])
    mean = embs.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=str)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--image_dir", type=Path)
    ap.add_argument("--eye", choices=["L", "R"], required=False)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove", type=str)
    ap.add_argument("--password", type=str, default=None)
    args = ap.parse_args()

    CFG.ensure_dirs()
    gallery = load_gallery()

    if args.list:
        print(f"Registered identities ({len(gallery['users'])}):")
        for uid, rec in gallery["users"].items():
            print(f"  {uid:20s}  eye={rec.get('eye', '?'):1s}  created={rec.get('created_at', '?')}")
        return

    if args.remove:
        if args.remove in gallery["users"]:
            del gallery["users"][args.remove]
            save_gallery(gallery)
            print(f"Removed {args.remove}")
        else:
            print(f"Not found: {args.remove}")
        return

    if not args.user_id or (not args.image and not args.image_dir):
        ap.error("--user_id and (--image | --image_dir) are required")

    device = get_device()
    model = load_model(device)

    if args.image_dir:
        emb = average_embeddings(model, device, args.image_dir)
    else:
        emb = embed_image(model, device, args.image)

    duplicate_of = None
    for uid, rec in gallery["users"].items():
        other = rec["embedding"]
        if isinstance(other, dict):
            if not args.password:
                continue
            other = decrypt_embedding(other, derive_key(args.password, bytes.fromhex(rec["salt"])))
        sim = float(np.dot(emb, other))
        if sim > CFG.duplicate_threshold:
            duplicate_of = (uid, sim)
            break

    if duplicate_of:
        print(f"Rejected: matches '{duplicate_of[0]}' (sim={duplicate_of[1]:.4f})")
        return

    record = {
        "embedding": emb,
        "eye": args.eye,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if CFG.cancelable_templates:
        seed = secrets.randbits(31)
        Q = random_orthogonal(CFG.embedding_dim, seed)
        record["embedding"] = apply_projection(emb, Q)
        record["projection_seed"] = seed
        record["cancelable"] = True

    if CFG.encryption_enabled:
        if not args.password:
            ap.error("Encryption is enabled — supply --password")
        salt = secrets.token_bytes(16)
        key = derive_key(args.password, salt)
        record["salt"] = salt.hex()
        record["embedding"] = encrypt_embedding(record["embedding"], key)
        record["encrypted"] = True

    gallery["users"][args.user_id] = record
    save_gallery(gallery)
    print(f"Registered '{args.user_id}'  (gallery size={len(gallery['users'])})")


if __name__ == "__main__":
    main()
