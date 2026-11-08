"""Gradio web interface."""
import json

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from register import load_gallery, load_model, save_gallery
from src.dataset import build_eval_transform, preprocess_iris
from src.liveness import estimate_liveness
from src.security import decrypt_embedding, derive_key
from src.utils import get_device

CFG.ensure_dirs()
DEVICE = get_device()
MODEL = load_model(DEVICE)


@torch.no_grad()
def _embed(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
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
    for uid, rec in gallery["users"].items():
        other = rec["embedding"]
        if isinstance(other, dict):
            continue
        sim = float(np.dot(emb, other))
        if sim > CFG.duplicate_threshold:
            return f"❌ Duplicate of '{uid}' (sim={sim:.4f})"
    gallery["users"][user_id] = {"embedding": emb, "eye": eye}
    save_gallery(gallery)
    return f"✅ Registered '{user_id}'  (liveness={live:.3f})"


def cb_verify(user_id, image, password):
    if not user_id or image is None:
        return "❌ Provide an ID and an image.", None
    gallery = load_gallery()
    if user_id not in gallery["users"]:
        return f"❌ Unknown user '{user_id}'", None
    rec = gallery["users"][user_id]
    ref = rec["embedding"]
    if isinstance(ref, dict):
        if not password:
            return "❌ Encrypted identity — supply password.", None
        ref = decrypt_embedding(ref, derive_key(password, bytes.fromhex(rec["salt"])))
    probe = _embed(image)
    sim = float(np.dot(probe, ref))
    ok = sim >= CFG.verify_threshold
    verdict = "✅ ACCEPT" if ok else "❌ REJECT"
    return (f"{verdict}   sim={sim:.4f}   threshold={CFG.verify_threshold}", sim)


def cb_identify(image, top_k, password):
    if image is None:
        return []
    probe = _embed(image)
    gallery = load_gallery()
    scores = []
    for uid, rec in gallery["users"].items():
        ref = rec["embedding"]
        if isinstance(ref, dict):
            if not password:
                continue
            try:
                ref = decrypt_embedding(ref, derive_key(password, bytes.fromhex(rec["salt"])))
            except Exception:
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
    lines.append(f"**Embedding dimension:** {CFG.embedding_dim}")
    lines.append(f"**ArcFace:** s={CFG.arcface_s}, m={CFG.arcface_m} rad ({CFG.arcface_m * 180 / np.pi:.1f}°)")
    lines.append(f"**Verification threshold:** {CFG.verify_threshold}")
    lines.append(f"**Duplicate threshold:** {CFG.duplicate_threshold}")
    gallery = load_gallery()
    lines.append(f"**Registered identities:** {len(gallery['users'])}")
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
