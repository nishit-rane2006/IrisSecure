"""Pure implementations of each pipeline stage."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from src.dataset import IrisDataset, build_eval_transform, build_train_transform, preprocess_iris
from src.metrics import compute_biometric_metrics
from src.model import IrisEmbeddingNet
from src.utils import EMA, get_device, optimize_cpu_runtime, save_json, set_seed


# ======================================================================
# Stage 0 — Clean
# ======================================================================
def _image_hash(img: np.ndarray, size: int = 16) -> bytes:
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = (resized > mean).flatten()
    return np.packbits(bits).tobytes()


def _hamming(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def _derive_label(subject_id: int, eye: str) -> int:
    return subject_id * 2 + (0 if eye.upper() == "L" else 1)


def clean_impl(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    root = CFG.data_root
    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    valid_rows = []
    rejected = {
        "corrupt": [],
        "wrong_resolution": [],
        "duplicate": [],
        "malformed_path": [],
    }
    seen_hashes: dict[int, list[tuple[str, bytes]]] = defaultdict(list)

    min_res = CFG.clean_min_resolution
    max_res = CFG.clean_max_resolution
    dup_thresh = CFG.clean_duplicate_threshold

    all_paths: list[tuple[Path, int, str]] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue
        for eye_dir in subject_dir.iterdir():
            if not eye_dir.is_dir():
                continue
            eye = eye_dir.name.upper()
            if eye not in ("L", "R"):
                continue
            for img_path in eye_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_paths.append((img_path, subject_id, eye))

    print(f"[clean] scanning {len(all_paths)} candidate files…")

    for img_path, subject_id, eye in tqdm(all_paths, desc="clean"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected["corrupt"].append(str(img_path))
            continue

        h, w = img.shape[:2]
        if h < min_res or w < min_res or h > max_res or w > max_res:
            rejected["wrong_resolution"].append({"path": str(img_path), "shape": [h, w]})
            continue

        label = _derive_label(subject_id, eye)
        phash = _image_hash(img)

        is_duplicate = False
        for existing_path, existing_hash in seen_hashes[label]:
            if _hamming(phash, existing_hash) <= dup_thresh:
                rejected["duplicate"].append({"path": str(img_path), "duplicate_of": existing_path})
                is_duplicate = True
                break
        if is_duplicate:
            continue

        seen_hashes[label].append((str(img_path), phash))
        valid_rows.append({
            "path": str(img_path),
            "label": label,
            "subject": subject_id,
            "eye": eye,
            "height": h,
            "width": w,
        })

    df = pd.DataFrame(valid_rows)

    counts = df["label"].value_counts() if len(df) else pd.Series(dtype=int)
    too_small = counts[counts < CFG.clean_min_images_per_class].index.tolist()
    if too_small:
        for lab in too_small:
            rows = df[df["label"] == lab]
            for _, r in rows.iterrows():
                rejected.setdefault("single_image_class", []).append(str(r["path"]))
        df = df[~df["label"].isin(too_small)].reset_index(drop=True)

    df.to_csv(out_dir / "clean.csv", index=False)

    report = {
        "total_candidates": len(all_paths),
        "kept": len(df),
        "rejected_counts": {k: len(v) for k, v in rejected.items()},
        "num_classes_kept": int(df["label"].nunique()) if len(df) else 0,
        "num_classes_dropped": len(too_small),
        "rejected_examples": {k: v[:10] for k, v in rejected.items()},
        "resolution_summary": {
            "min": [int(df["height"].min()), int(df["width"].min())] if len(df) else None,
            "max": [int(df["height"].max()), int(df["width"].max())] if len(df) else None,
            "mean": [float(df["height"].mean()), float(df["width"].mean())] if len(df) else None,
        },
    }
    save_json(report, out_dir / "cleaning_report.json")

    print(f"[clean] kept {report['kept']} images across {report['num_classes_kept']} classes")
    for k, v in report["rejected_counts"].items():
        if v:
            print(f"[clean]   rejected {v:5d}  ({k})")


# ======================================================================
# Stage 1 — Preprocess
# ======================================================================
def preprocess_impl(out_dir: Path):
    set_seed(CFG.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = CFG.work_dir / "artifacts" / "clean" / "clean.csv"
    if not clean_csv.exists():
        raise RuntimeError(f"Clean manifest not found at {clean_csv}. Run the clean stage first.")

    df = pd.read_csv(clean_csv)
    df["path"] = df["path"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} clean images across {df['label'].nunique()} classes")

    train_df, temp_df = train_test_split(
        df,
        test_size=CFG.val_ratio + CFG.test_ratio,
        stratify=df["label"],
        random_state=CFG.seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=CFG.test_ratio / (CFG.val_ratio + CFG.test_ratio),
        stratify=temp_df["label"],
        random_state=CFG.seed,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    classes = sorted(df["label"].unique())
    pd.Series({i: c for i, c in enumerate(classes)}).to_csv(out_dir / "label_map.csv")
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    eda = out_dir / "eda"
    eda.mkdir(exist_ok=True)
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    plt.hist(counts.values, bins=30)
    plt.title("Images per class (after cleaning)")
    plt.tight_layout()
    plt.savefig(eda / "images_per_class.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sdf, name in zip(axes, [train_df, val_df, test_df], ["train", "val", "test"]):
        sc = sdf["label"].value_counts().sort_index()
        ax.plot(sc.index, sc.values)
        ax.set_title(f"{name} (n={len(sdf)})")
    plt.tight_layout()
    plt.savefig(eda / "splits.png", dpi=140)
    plt.close()


# ======================================================================
# Stage 2 — Train
# ======================================================================
@torch.no_grad()
def _evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict_logits(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def _build_optimizer(model, cfg, phase):
    if phase == "A":
        params = list(model.head.parameters()) + list(model.arcface.parameters())
        return torch.optim.AdamW(params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.head_lr * cfg.backbone_lr_scale},
            {"params": model.head.parameters(), "lr": cfg.head_lr},
            {"params": model.arcface.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def train_impl(out_dir: Path, epochs: int = None, batch_size: int = None, grad_accum: int = None, num_workers: int = None, max_classes: int = None):
    epochs = epochs or CFG.epochs
    batch_size = batch_size or CFG.batch_size
    grad_accum = grad_accum or CFG.grad_accum_steps
    num_workers = num_workers if num_workers is not None else CFG.num_workers
    max_classes = max_classes if max_classes is not None else CFG.max_classes

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(CFG.seed)
    device = get_device()
    if device.type == "cpu":
        optimize_cpu_runtime(16)
    print(f"Device: {device}")

    pre_dir = CFG.work_dir / "artifacts" / "preprocess"
    train_df = pd.read_csv(pre_dir / "train.csv")
    val_df = pd.read_csv(pre_dir / "val.csv")

    if max_classes and max_classes > 0:
        keep = sorted(train_df["label"].unique())[:max_classes]
        train_df = train_df[train_df["label"].isin(keep)].reset_index(drop=True)
        val_df = val_df[val_df["label"].isin(keep)].reset_index(drop=True)
        print(f"Subsampled to {len(keep)} classes")

    train_ds = IrisDataset(
        list(zip(train_df["path"], train_df["label"])),
        transform=build_train_transform(CFG.image_size),
        size=CFG.image_size,
    )
    val_ds = IrisDataset(
        list(zip(val_df["path"], val_df["label"])),
        transform=build_eval_transform(CFG.image_size),
        size=CFG.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    num_classes = int(train_df["label"].max()) + 1
    print(f"num_classes = {num_classes}")

    model = IrisEmbeddingNet(
        num_classes=num_classes,
        embedding_dim=CFG.embedding_dim,
        dropout=CFG.dropout,
        arcface_s=CFG.arcface_s,
        arcface_m=CFG.arcface_m,
        label_smoothing=CFG.label_smoothing,
        pretrained=CFG.pretrained,
    ).to(device)
    if CFG.channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_amp = CFG.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = max(1, grad_accum)

    model.set_backbone_trainable(False)
    optimizer = _build_optimizer(model, CFG, "A")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=CFG.head_lr * 0.01,
    )
    ema = EMA(model, decay=CFG.ema_decay)

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == CFG.warmup_epochs + 1:
            print("\n>> Unfreezing backbone (Phase B)")
            model.set_backbone_trainable(True)
            optimizer = _build_optimizer(model, CFG, "B")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=CFG.head_lr * 0.01,
            )
            ema = EMA(model, decay=CFG.ema_decay)

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(pbar):
            if CFG.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=use_amp):
                _, loss, _ = model(images, labels)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            running_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val_acc = _evaluate_acc(model, val_loader, device)
        model.load_state_dict(backup)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_acc={val_acc:.4%}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "num_classes": num_classes,
                    "embedding_dim": CFG.embedding_dim,
                    "val_acc": val_acc,
                    "arcface_s": CFG.arcface_s,
                    "arcface_m": CFG.arcface_m,
                },
                out_dir / "best_model.pth",
            )
            print(f"  >> saved best_model.pth  (val_acc={val_acc:.4%})")

    save_json({"best_val_acc": best_acc, "history": history}, out_dir / "train_history.json")
    print(f"\nDone. Best val acc = {best_acc:.4%}")


# ======================================================================
# Stage 3 — Embed
# ======================================================================
@torch.no_grad()
def embed_impl(out_dir: Path, batch_size: int = None, num_workers: int = None, tta: bool = True):
    batch_size = batch_size or CFG.batch_size
    num_workers = num_workers if num_workers is not None else CFG.num_workers
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    ckpt = torch.load(CFG.work_dir / "artifacts" / "train" / "best_model.pth", map_location="cpu")
    model = IrisEmbeddingNet(ckpt["num_classes"], ckpt["embedding_dim"], pretrained=False).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]), strict=False)
    if CFG.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    print(f"Loaded checkpoint (val_acc={ckpt['val_acc']:.4%})")

    pre_dir = CFG.work_dir / "artifacts" / "preprocess"
    dfs = [pd.read_csv(pre_dir / f"{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(dfs, ignore_index=True)

    ds = IrisDataset(
        list(zip(df["path"], df["label"])),
        transform=build_eval_transform(CFG.image_size),
        size=CFG.image_size,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    embs, labels = [], []
    for images, ys in tqdm(loader, desc="embedding"):
        images = images.to(device, non_blocking=True)
        e = model.embed(images)
        if tta:
            e = F.normalize(e + model.embed(torch.flip(images, dims=[3])), dim=1)
        embs.append(e.cpu().numpy().astype(np.float32))
        labels.append(ys.numpy())

    np.savez_compressed(
        out_dir / "embeddings.npz",
        embeddings=np.concatenate(embs),
        labels=np.concatenate(labels),
        paths=np.array(df["path"].tolist()),
    )
    print(f"Saved {sum(len(e) for e in embs)} embeddings")


# ======================================================================
# Stage 4 — Evaluate
# ======================================================================
def _generate_pairs(labels: np.ndarray, max_impostor: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    genuine = []
    for idxs in by_class.values():
        if len(idxs) < 2:
            continue
        for a, b in combinations(idxs, 2):
            genuine.append((a, b))

    n = len(labels)
    impostor = set()
    attempts = 0
    while len(impostor) < max_impostor and attempts < max_impostor * 20:
        a, b = rng.integers(0, n, size=2)
        attempts += 1
        if a == b or labels[a] == labels[b]:
            continue
        impostor.add((min(a, b), max(a, b)))
    return genuine, list(impostor)


def evaluate_impl(out_dir: Path, split: str = "test", max_impostor: int = None):
    max_impostor = max_impostor or CFG.max_impostor_pairs
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(CFG.work_dir / "artifacts" / "embed" / "embeddings.npz", allow_pickle=True)
    embs, labels, paths = data["embeddings"], data["labels"], data["paths"]

    split_df = pd.read_csv(CFG.work_dir / "artifacts" / "preprocess" / f"{split}.csv")
    split_paths = set(split_df["path"].tolist())
    mask = np.array([p in split_paths for p in paths])
    embs, labels = embs[mask], labels[mask]
    print(f"Evaluating on {len(embs)} {split} embeddings")

    genuine_idx, impostor_idx = _generate_pairs(labels, max_impostor, CFG.seed)
    print(f"Genuine pairs: {len(genuine_idx)}  Impostor pairs: {len(impostor_idx)}")

    g_scores = np.array([float(np.dot(embs[a], embs[b])) for a, b in genuine_idx])
    i_scores = np.array([float(np.dot(embs[a], embs[b])) for a, b in impostor_idx])

    metrics = compute_biometric_metrics(g_scores, i_scores)
    printable = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    save_json(printable, out_dir / "report.json")

    for k, v in printable.items():
        print(f"  {k}: {v}")

    fpr = np.array(metrics["fpr"])
    tpr = np.array(metrics["tpr"])
    fnr = 1 - tpr
    thr = np.array(metrics["thresholds"])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xscale("log")
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.title(f"ROC (AUC = {metrics['roc_auc']:.6f})")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, fnr, lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET curve")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "det.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(g_scores, bins=80, alpha=0.6, label="Genuine", color="tab:green")
    plt.hist(i_scores, bins=80, alpha=0.6, label="Impostor", color="tab:red")
    plt.axvline(metrics["eer_threshold"], color="k", ls="--", label=f"EER thr = {metrics['eer_threshold']:.4f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("count")
    plt.title("Score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "scores.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thr, fpr, label="FAR")
    plt.plot(thr, fnr, label="FRR")
    plt.xlabel("threshold")
    plt.ylabel("rate")
    plt.title("FAR / FRR vs threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "far_frr.png", dpi=150)
    plt.close()

    print(f"\nSaved plots + report -> {out_dir}")
