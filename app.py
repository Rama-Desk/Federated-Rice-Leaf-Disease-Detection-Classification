from pathlib import Path
import json

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model import build_model

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
CONFIG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
CLASS_NAMES = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
SIZE = int(CONFIG["input_size"][0])
MEAN, STD = CONFIG["normalization"]["mean"], CONFIG["normalization"]["std"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="FedRice-XAI | Rice Leaf Assistant", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --green:#176B4D; --deep:#103D30; --mint:#EAF5EF; --ink:#172B24; --muted:#64756D; --line:#E2EAE5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(180deg,#F7FBF8 0%,#FFFFFF 42%); }
.block-container { max-width:1120px; padding-top:1.6rem; padding-bottom:3rem; }
.hero { background:linear-gradient(115deg,#103D30 0%,#176B4D 62%,#2B8764 100%); border-radius:24px; padding:30px 34px; color:white; box-shadow:0 14px 36px rgba(16,61,48,.13); }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(1.8rem,4vw,2.65rem); font-weight:800; margin:0 0 8px 0; color:#fff; letter-spacing:-.04em; }
.hero p { color:#E1F1E8; font-size:1.02rem; margin:0; max-width:720px; line-height:1.6; }
.pill { display:inline-block; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.24); padding:6px 11px; border-radius:999px; font-size:.78rem; font-weight:700; margin-bottom:14px; }
.section-title { font-family:Manrope,sans-serif; font-weight:800; font-size:1.25rem; letter-spacing:-.02em; margin:1.5rem 0 .25rem; }
.subtle { color:var(--muted); font-size:.94rem; line-height:1.55; }
.panel { background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 5px 18px rgba(22,56,40,.045); }
.result { background:linear-gradient(135deg,#F0F8F3,#FFFFFF); border:1px solid #D7E9DD; border-radius:18px; padding:22px; }
.result-label { color:#527163; text-transform:uppercase; font-size:.72rem; letter-spacing:.09em; font-weight:800; }
.result-name { color:var(--deep); font-family:Manrope,sans-serif; font-size:1.75rem; font-weight:800; line-height:1.2; margin:.35rem 0 .65rem; }
.notice { background:#FFF8E8; border:1px solid #F0DFB8; border-radius:14px; padding:14px 16px; color:#634A16; font-size:.9rem; line-height:1.5; }
.good-note { background:#EFF7F1; border:1px solid #D7E9DD; border-radius:14px; padding:14px 16px; color:#315D45; font-size:.9rem; line-height:1.5; }
.stButton>button { border-radius:12px; min-height:3rem; font-weight:700; }
[data-testid="stFileUploader"] { background:#fff; border:1.5px dashed #B8D3C2; border-radius:16px; padding:12px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
footer {visibility:hidden;} #MainMenu {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the rice-leaf model…")
def load_model():
    model = build_model(len(CLASS_NAMES)).to(DEVICE)
    weights = torch.load(MODEL_DIR / CONFIG["primary_prediction_model"], map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

FRIENDLY = {
    "bacterial_leaf_blight": ("Bacterial leaf blight", "A bacterial disease that can cause water-soaked lesions and drying that spreads from leaf tips or edges."),
    "bacterial_leaf_streak": ("Bacterial leaf streak", "A bacterial disease associated with narrow, water-soaked streaks that may become translucent or brown."),
    "bacterial_panicle_blight": ("Bacterial panicle blight", "A bacterial disease affecting the panicle and grain development. A leaf-only photo may not show its key symptoms."),
    "blast": ("Rice blast", "A fungal disease that can produce spindle-shaped lesions with pale centres and darker margins."),
    "brown_spot": ("Brown spot", "A fungal disease often associated with oval brown lesions, sometimes with a yellow halo."),
    "dead_heart": ("Dead heart", "A symptom in which the central shoot of a young rice tiller dries and can be pulled out easily."),
    "downy_mildew": ("Downy mildew", "A disease that can cause pale or yellowish streaking and abnormal leaf growth."),
    "hispa": ("Rice hispa damage", "Feeding or scraping damage caused by the rice hispa beetle; leaves may show white streaks or scraped patches."),
    "normal": ("No disease pattern detected", "The model assigned this image to its normal-leaf class. This does not prove the plant is disease-free."),
    "tungro": ("Rice tungro", "A viral disease complex that may cause yellow-orange leaf discoloration and stunted plant growth."),
}


def mc_dropout(model, x, passes=25):
    # Enable only dropout modules for stochastic forward passes; keep batch norm in eval mode.
    model.eval()
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            dropout_layers.append(module)
    outputs = []
    with torch.no_grad():
        for _ in range(passes):
            outputs.append(torch.softmax(model(x), dim=1).cpu().numpy()[0])
    model.eval()
    arr = np.asarray(outputs)
    return arr.mean(axis=0), arr.std(axis=0)


def make_gradcam(model, x, class_idx):
    activations, gradients = {}, {}
    target = model.features[-1]
    h1 = target.register_forward_hook(lambda _m, _i, out: activations.setdefault("v", out))
    def grad_hook(_m, _gi, go):
        gradients["v"] = go[0]
    h2 = target.register_full_backward_hook(grad_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
        weights = gradients["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["v"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    finally:
        h1.remove(); h2.remove(); model.zero_grad(set_to_none=True); model.eval()


def overlay_cam(image, cam):
    base = image.resize((SIZE, SIZE)).convert("RGB")
    heat = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    heat[..., 0] = np.uint8(cam * 255)
    heat[..., 1] = np.uint8(np.clip(cam - .35, 0, 1) * 90)
    heat_img = Image.fromarray(heat).resize(base.size)
    return Image.blend(base, heat_img, 0.38)


def friendly(name):
    return FRIENDLY.get(name, (name.replace("_", " ").title(), "The model's predicted class."))

st.markdown("""
<div class="hero">
  <div class="pill">🌾 AI-POWERED RICE LEAF ANALYSIS</div>
  <h1>Understand your rice leaf health</h1>
  <p>Upload a clear photo of a rice leaf to see the model's most likely class, a visual explanation, and guidance on how to interpret the result.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">📷 Upload a leaf photo</div>', unsafe_allow_html=True)
st.markdown('<p class="subtle">For a more useful result, photograph one leaf in natural light. Keep the leaf in focus and make sure symptoms are visible.</p>', unsafe_allow_html=True)
col_upload, col_tips = st.columns([1.35, .85], gap="large")
with col_upload:
    uploaded = st.file_uploader("Choose a JPG, JPEG, PNG or WEBP image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with col_tips:
    st.markdown("""
    <div class="panel">
      <b>📸 Photo tips</b><br><br>
      <span class="subtle">• Capture the whole affected leaf<br>• Avoid shadows and motion blur<br>• Keep other leaves out of the frame<br>• Do not use screenshots or edited images</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="notice">⚠️ <b>Important limitation:</b> This version classifies images into the 10 rice-leaf classes it was trained on. It does not yet have a validated rice-vs-non-rice / out-of-distribution rejection gate. A non-rice or unfamiliar image may therefore receive a misleading rice-class prediction. Use rice-leaf photos only and confirm important decisions with an agricultural expert.</div>', unsafe_allow_html=True)

if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        if min(image.size) < 80:
            st.error("This image is too small to analyse. Please upload a clearer, higher-resolution photo.")
        else:
            st.markdown('<div class="section-title">🔍 Analysis</div>', unsafe_allow_html=True)
            with st.spinner("Analysing the image…"):
                model = load_model()
                x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
                mean_probs, std_probs = mc_dropout(model, x, passes=25)
                pred_idx = int(np.argmax(mean_probs))
                cam = make_gradcam(model, x, pred_idx)
                pred_key = CLASS_NAMES[pred_idx]
                pred_title, pred_desc = friendly(pred_key)
                score = float(mean_probs[pred_idx])
                variability = float(std_probs[pred_idx])

            image_col, result_col = st.columns([.9, 1.1], gap="large")
            with image_col:
                st.markdown('<div class="panel"><b>🖼️ Your uploaded image</b></div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with result_col:
                st.markdown(f'<div class="result"><div class="result-label">Model prediction · most likely class</div><div class="result-name">{pred_title}</div><div style="color:#486456;line-height:1.55">{pred_desc}</div></div>', unsafe_allow_html=True)
                st.metric("Model probability score", f"{score:.1%}")
                st.caption("This score is the model's output probability, not a calibrated measure of diagnostic certainty.")
                if variability >= 0.08:
                    st.warning("The model's outputs varied across repeated dropout passes. Treat this result as uncertain and seek expert confirmation.")
                else:
                    st.info("The model showed relatively low variation across repeated passes. This still does not confirm that the prediction is correct or that the image is a rice leaf.")

            st.markdown('<div class="section-title">🧠 Why did the model predict this?</div>', unsafe_allow_html=True)
            cam_col1, cam_col2 = st.columns([1, 1], gap="large")
            with cam_col1:
                st.image(overlay_cam(image, cam), caption="Grad-CAM overlay — highlighted regions influenced the model output", use_container_width=True)
            with cam_col2:
                st.markdown("**How to read this explanation**")
                st.markdown("The coloured overlay shows image regions that contributed to the selected class score. It is an explanation aid—not a disease boundary, laboratory test, or proof that the model focused on the correct symptom.")
                st.markdown("**What to do next**")
                st.markdown("1. Compare the visible symptoms with reliable local agricultural guidance.\n2. Take another clear photo if the leaf is damaged, shadowed, or out of focus.\n3. Ask a Krishi Vigyan Kendra (KVK) or agriculture officer before applying treatment.")

            with st.expander("See the next most likely model classes"):
                st.caption("These are alternative model scores, not confirmed diagnoses.")
                order = np.argsort(mean_probs)[::-1]
                for idx in order[:3]:
                    title, _ = friendly(CLASS_NAMES[int(idx)])
                    st.write(f"**{title}** — {mean_probs[idx]:.1%}")
                    st.progress(float(mean_probs[idx]))

            st.markdown('<div class="good-note">🌱 <b>Responsible use:</b> This tool is a research prototype for decision support. It should not be used alone to choose pesticides, fungicides, or crop-treatment actions.</div>', unsafe_allow_html=True)
    except Exception as exc:
        st.error("The image could not be analysed. Please try another JPG, PNG or WEBP image.")
        with st.expander("Technical details"):
            st.code(str(exc))
else:
    st.markdown('<div class="panel" style="text-align:center;padding:30px;margin-top:14px"><div style="font-size:2.2rem">🌿</div><b>Your result will appear here</b><div class="subtle">Upload a clear rice-leaf photo to begin the analysis.</div></div>', unsafe_allow_html=True)

with st.expander("ℹ️ About this research model"):
    st.markdown("**Architecture:** MobileNetV3-Small · **Input:** 224 × 224 pixels · **Classes:** 10 rice-leaf categories · **Model:** centralized Dataset 2 checkpoint.")
    st.markdown("The supplied research notebook reports **96.41% test accuracy** and **96.22% test Macro-F1** on its held-out Dataset 2 test split. These are experimental results on that dataset—not a guarantee for new farms, cameras, lighting conditions, or unfamiliar images.")
    st.markdown("**PlantVillage:** It is used in the research workflow as external non-rice / OOD data for evaluation. The deployed checkpoint has not yet been shown to pass a validated OOD rejection test. Do not interpret this application as having reliable non-rice image detection until that evaluation and threshold calibration are completed.")
    st.markdown("**Model variability:** Repeated dropout passes provide a variability indicator only. The displayed probability and variability are not calibrated diagnostic certainty.")

st.markdown("<hr style='border:0;border-top:1px solid #E2EAE5;margin:2rem 0 1rem'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#718078;font-size:.82rem'>FedRice-XAI · Research prototype · Built for explainable rice-leaf image classification</div>", unsafe_allow_html=True)
