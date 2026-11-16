"""Gradio web interface for prepared IrisSecure and IrisGuard artifacts."""
import json
import secrets
from datetime import datetime, timezone

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CFG
from config.hardware_config import configure_torch, torch_device
from irisguard.afsdd import AFSDD, AFSDDConfig
from irisguard.artifact_manager import ArtifactError, ArtifactManager
from irisguard.sqlite_store import SQLiteTemplateStore
from register import load_gallery, load_model, save_gallery
from scripts.common import preprocessing_config
from src.dataset import build_eval_transform, preprocess_iris
from src.liveness import estimate_liveness
from src.quality import estimate_image_quality
from src.security import decrypt_embedding, derive_key, encrypt_embedding

CFG.ensure_dirs()


def _initialize_runtime():
    hardware = configure_torch()
    manager = ArtifactManager()
    device = torch_device(hardware)
    model = load_model(device)
    model_hash = str(manager.read_manifest()["artifacts"]["model"]["sha256"])
    dataset_fingerprint = manager.sha256(CFG.work_dir / "artifacts" / "clean" / "clean.csv")
    preprocessing_hash = manager.fingerprint_json(preprocessing_config())
    required = ("model", "embeddings", "embedding_labels", "calibration")
    missing = []
    prepared = {}
    compatibility = {
        "embeddings": (model_hash, dataset_fingerprint, preprocessing_hash),
        "embedding_labels": (model_hash, dataset_fingerprint, preprocessing_hash),
        "calibration": (model_hash, dataset_fingerprint, preprocessing_hash),
    }
    for name in required[1:]:
        try:
            prepared[name] = manager.validate(
                name,
                model_hash=compatibility[name][0],
                dataset_fingerprint=compatibility[name][1],
                preprocessing_hash=compatibility[name][2],
            )
        except ArtifactError:
            missing.append(name)
    if missing:
        commands = {
            "model": "python .\\scripts\\train_once.py",
            "embeddings": "python .\\scripts\\extract_embeddings.py",
            "embedding_labels": "python .\\scripts\\extract_embeddings.py",
            "calibration": "python .\\scripts\\calibrate_once.py",
        }
        todo = []
        for name in missing:
            if commands[name] not in todo:
                todo.append(commands[name])
        raise SystemExit(
            "Required prepared artifacts are missing or invalid: "
            + ", ".join(missing)
            + ". Run, in order: "
            + "\n"
            + "\n".join(todo)
        )

    with prepared["calibration"].open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    CFG.verify_threshold = float(calibration["verify_threshold"])
    embeddings = np.load(prepared["embeddings"], mmap_mode="r")
    labels = pd.read_csv(prepared["embedding_labels"])
    if embeddings.ndim != 2 or embeddings.shape[1] != CFG.embedding_dim or len(embeddings) != len(labels):
        raise SystemExit("Prepared embedding matrix and label metadata are incompatible; rerun scripts/extract_embeddings.py.")
    store = SQLiteTemplateStore(CFG.template_db_path)
    if hardware.available_ram_bytes is None:
        print(f"[IrisGuard] backend={hardware.selected_backend}, cpu={hardware.cpu_model}, threads={hardware.cpu_threads}")
    else:
        print(
            f"[IrisGuard] backend={hardware.selected_backend}, cpu={hardware.cpu_model}, "
            f"threads={hardware.cpu_threads}, free RAM={hardware.available_ram_bytes / (1024 ** 3):.1f} GiB"
        )
    return hardware, device, model, calibration, embeddings, store


HARDWARE, DEVICE, MODEL, CALIBRATION, EMBEDDINGS, TEMPLATE_STORE = _initialize_runtime()
_AFSDD_CONFIG_FIELDS = AFSDDConfig.__dataclass_fields__.keys()
AFSDD_CONFIG = AFSDDConfig(
    **{key: value for key, value in CALIBRATION["afsdd"].items() if key in _AFSDD_CONFIG_FIELDS}
)
_REFERENCE_STATS = CALIBRATION["afsdd"].get("reference_stats", {})


@torch.inference_mode()
def _embed(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
    img = preprocess_iris(gray, CFG.image_size)
    tf = build_eval_transform(CFG.image_size)
    t = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
    e = MODEL.embed(t)
    e = F.normalize(e + MODEL.embed(torch.flip(t, dims=[3])), dim=1)
    return e.squeeze(0).cpu().numpy().astype(np.float32)


def cb_register(user_id, image, eye, password):
    if not user_id or image is None:
        return "❌ Provide a user ID and an image."
    live = estimate_liveness(image)
    if live < 0.35:
        return f"❌ Liveness check failed (score={live:.3f})."
    emb = _embed(image)
    gallery = load_gallery()
    if user_id in gallery["users"]:
        return f"❌ User ID '{user_id}' is already registered."
    try:
        TEMPLATE_STORE.get(user_id)
        return f"❌ User ID '{user_id}' is already registered."
    except KeyError:
        pass

    for record in TEMPLATE_STORE.list_templates():
        sim = float(np.dot(emb, record["current_template"]))
        if sim > CFG.duplicate_threshold:
            return f"❌ Duplicate of '{record['user_id']}' (sim={sim:.4f})"
    for uid, rec in gallery["users"].items():
        other = rec["embedding"]
        if isinstance(other, dict):
            if not password:
                continue
            other = decrypt_embedding(other, derive_key(password, bytes.fromhex(rec["salt"])))
        sim = float(np.dot(emb, other))
        if sim > CFG.duplicate_threshold:
            return f"❌ Duplicate of '{uid}' (sim={sim:.4f})"

    if CFG.encryption_enabled:
        if not password:
            return "❌ Encryption is enabled — supply a password."
        salt = secrets.token_bytes(16)
        key = derive_key(password, salt)
        gallery["users"][user_id] = {
            "embedding": encrypt_embedding(emb, key),
            "salt": salt.hex(),
            "encrypted": True,
            "eye": eye,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_gallery(gallery)
        return f"✅ Registered '{user_id}' (encrypted legacy gallery; adaptive updates unavailable)"

    TEMPLATE_STORE.register(user_id, emb, quality=estimate_image_quality(image))
    gallery["users"][user_id] = {
        "embedding": emb,
        "eye": eye,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_gallery(gallery)
    return f"✅ Registered '{user_id}'  (liveness={live:.3f})"


def cb_verify(user_id, image, password):
    if not user_id or image is None:
        return "❌ Provide an ID and an image.", None
    probe = _embed(image)
    try:
        record = TEMPLATE_STORE.get(user_id)
    except KeyError:
        gallery = load_gallery()
        if user_id not in gallery["users"]:
            return f"❌ Unknown user '{user_id}'", None
        rec = gallery["users"][user_id]
        ref = rec["embedding"]
        if isinstance(ref, dict):
            if not password:
                return "❌ Encrypted identity — supply password.", None
            ref = decrypt_embedding(ref, derive_key(password, bytes.fromhex(rec["salt"])))
        sim = float(np.dot(probe, ref))
        ok = sim >= CFG.verify_threshold
        verdict = "✅ ACCEPT" if ok else "❌ REJECT"
        return (f"{verdict}   sim={sim:.4f}   threshold={CFG.verify_threshold}", sim)

    sim = float(np.dot(probe, record["current_template"]))
    ok = sim >= CFG.verify_threshold
    verdict = "✅ ACCEPT" if ok else "❌ REJECT"
    update_decision = "NOT_UPDATED"
    if ok:
        detector = AFSDD(AFSDD_CONFIG)
        detector.reference_stats.update(_REFERENCE_STATS)
        update_result = TEMPLATE_STORE.update(
            user_id,
            probe,
            quality=estimate_image_quality(image),
            authenticated=True,
            detector=detector,
        )
        update_decision = str(update_result["decision"])
    return (
        f"{verdict}   sim={sim:.4f}   threshold={CFG.verify_threshold}   "
        f"template update={update_decision}",
        sim,
    )


def cb_identify(image, top_k, password):
    if image is None:
        return []
    probe = _embed(image)
    gallery = load_gallery()
    scores = [
        (record["user_id"], float(np.dot(probe, record["current_template"])))
        for record in TEMPLATE_STORE.list_templates()
    ]
    for uid, rec in gallery["users"].items():
        if any(existing_uid == uid for existing_uid, _ in scores):
            continue
        ref = rec["embedding"]
        if isinstance(ref, dict):
            if not password:
                continue
            try:
                ref = decrypt_embedding(ref, derive_key(password, bytes.fromhex(rec["salt"])))
            except (ValueError, KeyError):
                continue
        scores.append((uid, float(np.dot(probe, ref))))
    scores.sort(key=lambda x: -x[1])
    return [[uid, f"{s:.4f}", "✓" if s >= CFG.verify_threshold else ""] for uid, s in scores[: int(top_k)]]


def cb_compare(img1, img2):
    if img1 is None or img2 is None:
        return "❌ Provide both images."
    s = float(np.dot(_embed(img1), _embed(img2)))
    return f"Cosine similarity: {s:.4f}"


def cb_dashboard():
    lines = ["# IrisSecure — Dashboard", ""]
    lines.append(f"**Device:** `{DEVICE}`")
    lines.append(f"**Backend:** `{HARDWARE.selected_backend}`")
    lines.append(f"**CPU threads:** `{HARDWARE.cpu_threads}`")
    lines.append(f"**Embedding dimension:** {CFG.embedding_dim}")
    lines.append(f"**ArcFace:** s={CFG.arcface_s}, m={CFG.arcface_m} rad ({CFG.arcface_m * 180 / np.pi:.1f}°)")
    lines.append(f"**Verification threshold:** {CFG.verify_threshold}")
    lines.append(f"**Duplicate threshold:** {CFG.duplicate_threshold}")
    gallery = load_gallery()
    lines.append(
        f"**Registered identities:** "
        f"{len(set(gallery['users']) | {record['user_id'] for record in TEMPLATE_STORE.list_templates()})}"
    )
    report_path = CFG.work_dir / "artifacts" / "evaluate" / "report.json"
    if report_path.exists():
        with open(report_path) as f:
            r = json.load(f)
        lines += [
            "",
            "## Latest evaluation",
            f"- ROC AUC: `{r.get('roc_auc', 0):.6f}`",
            f"- EER: `{r.get('eer', 0):.5%}`",
            f"- TAR@FAR=0.001: `{r.get('tar_at_far_1e-3', 0):.4%}`",
        ]
    return "\n".join(lines)


def build_app():
    with gr.Blocks(title="IrisSecure", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛡️ IrisSecure\nIris-based biometric authentication — ResNet50 + ArcFace")
        with gr.Tabs():
            with gr.Tab("Register"):
                with gr.Row():
                    with gr.Column():
                        r_uid = gr.Textbox(label="User ID", placeholder="john_L")
                        r_eye = gr.Radio(["L", "R"], label="Eye", value="L")
                        r_img = gr.Image(label="Iris image", type="numpy")
                        r_pwd = gr.Textbox(label="Password (if encryption on)", type="password")
                        r_btn = gr.Button("Register", variant="primary")
                    r_out = gr.Textbox(label="Result", lines=3)
                r_btn.click(cb_register, [r_uid, r_img, r_eye, r_pwd], r_out)

            with gr.Tab("Verify (1:1)"):
                with gr.Row():
                    with gr.Column():
                        v_uid = gr.Textbox(label="User ID")
                        v_img = gr.Image(label="Probe image", type="numpy")
                        v_pwd = gr.Textbox(label="Password", type="password")
                        v_btn = gr.Button("Verify", variant="primary")
                    with gr.Column():
                        v_out = gr.Textbox(label="Result", lines=2)
                        v_sim = gr.Number(label="Similarity")
                v_btn.click(cb_verify, [v_uid, v_img, v_pwd], [v_out, v_sim])

            with gr.Tab("Identify (1:N)"):
                with gr.Row():
                    with gr.Column():
                        i_img = gr.Image(label="Probe image", type="numpy")
                        i_k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                        i_pwd = gr.Textbox(label="Password", type="password")
                        i_btn = gr.Button("Search gallery", variant="primary")
                    i_out = gr.Dataframe(headers=["User", "Similarity", "Match"], label="Ranked matches")
                i_btn.click(cb_identify, [i_img, i_k, i_pwd], i_out)

            with gr.Tab("Compare"):
                with gr.Row():
                    c1 = gr.Image(label="Image 1", type="numpy")
                    c2 = gr.Image(label="Image 2", type="numpy")
                c_btn = gr.Button("Compare", variant="primary")
                c_out = gr.Textbox(label="Result")
                c_btn.click(cb_compare, [c1, c2], c_out)

            with gr.Tab("Dashboard"):
                d_md = gr.Markdown()
                d_btn = gr.Button("Refresh")
                d_btn.click(cb_dashboard, None, d_md)
                demo.load(cb_dashboard, None, d_md)

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
