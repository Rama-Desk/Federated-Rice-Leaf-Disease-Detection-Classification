from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from model import build_model

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
CONFIG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
CLASS_NAMES = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
SIZE = int(CONFIG.get("input_size", [224, 224])[0])
MEAN = CONFIG["normalization"]["mean"]
STD = CONFIG["normalization"]["std"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="KrishiMitra | Rice Leaf Disease Classification", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--forest:#123d2b;--green:#167344;--mint:#edf7ef;--ink:#12352a;--muted:#64766c;--line:#e1ebe3;--gold:#d6a84f}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:#f7faf8}.block-container{max-width:1440px;padding-top:.7rem;padding-bottom:2.5rem}
header[data-testid="stHeader"]{background:transparent;height:2.2rem}
.brandbar{display:flex;align-items:center;gap:13px;padding:8px 4px 14px;border-bottom:1px solid #e5ece7;margin-bottom:14px}
.brandmark{width:42px;height:42px;border-radius:50%;background:#e8f5e9;display:flex;align-items:center;justify-content:center;font-size:25px}
.brandname{font-family:Manrope,sans-serif;font-weight:800;font-size:1.55rem;color:#123d2b;letter-spacing:-.04em}
.brandtag{font-size:.9rem;color:#53675d;border-left:1px solid #cbd8cf;padding-left:16px;margin-left:4px}
.hero-wrap{display:grid;grid-template-columns:48% 52%;min-height:350px;background:linear-gradient(110deg,#f5faf6,#eaf5ec);border:1px solid #e4eee6;border-radius:22px;overflow:hidden;margin:10px 0 20px}
.hero-copy{padding:40px 38px;display:flex;flex-direction:column;justify-content:center}
.hero-chip{display:inline-block;background:#e0f2e4;color:#155b35;padding:8px 14px;border-radius:30px;font-weight:700;font-size:.83rem;width:max-content;margin-bottom:17px}
.hero-title{font-family:Manrope,sans-serif;font-size:clamp(2rem,3.5vw,3.15rem);line-height:1.07;letter-spacing:-.045em;color:#103a2b;font-weight:800;margin:0 0 16px}
.hero-desc{font-size:1.05rem;line-height:1.65;color:#28483a;max-width:590px;margin:0 0 23px}
.hero-photo{min-height:350px;background-size:cover;background-position:center}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:19px 21px;box-shadow:0 5px 18px rgba(20,55,39,.035);height:100%}
.soft{background:#edf6f0;border:1px solid #d9e9de;border-radius:16px;padding:17px 19px}
.warn{background:#fff7e8;border:1px solid #efdfbd;border-radius:14px;padding:14px 16px;color:#674e1c;line-height:1.55}
.info{background:#eef4fb;border:1px solid #d9e5f3;border-radius:14px;padding:14px 16px;color:#34516c;line-height:1.55}
.kicker{font-size:.75rem;color:#698075;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.big{font-family:Manrope,sans-serif;font-weight:800;font-size:1.75rem;color:var(--forest);line-height:1.2}
.section{font-family:Manrope,sans-serif;font-weight:800;font-size:1.3rem;margin:1.35rem 0 .35rem;color:var(--forest)}
.muted{color:var(--muted);line-height:1.6;font-size:.93rem}
.stButton>button{border-radius:12px;font-weight:700;min-height:2.8rem;border:1px solid #dce9df}
.stButton>button[kind="primary"]{background:#08713c;border-color:#08713c;color:#fff}
[data-testid="stFileUploader"]{background:white;border:1.5px dashed #b8d2c1;border-radius:16px;padding:12px}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px}
div[role="radiogroup"]{gap:7px;justify-content:flex-end;flex-wrap:wrap}
div[role="radiogroup"] label{background:#fff;border:1px solid #e0e9e2;border-radius:10px;padding:7px 13px}
div[role="radiogroup"] label:has(input:checked){background:#e7f4e9;border-color:#c5e0cb;color:#155b35;font-weight:700}
footer{visibility:hidden}#MainMenu{visibility:hidden}
@media(max-width:800px){.hero-wrap{grid-template-columns:1fr}.hero-copy{padding:28px 23px}.hero-photo{min-height:210px}.brandtag{display:none}.brandname{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)

st.markdown('\n<style>\n:root{--forest:#073d2b;--green:#08763e;--mint:#e9f5eb;--ink:#092f23;--muted:#5d6b64;--line:#e2ebe5}\nhtml,body,[class*="css"]{font-family:\'DM Sans\',sans-serif;color:var(--ink)}\n.stApp{background:#f7faf8}\n.block-container{max-width:1600px;padding:0 2.2rem 2.2rem}\nheader[data-testid="stHeader"]{height:1.2rem;background:transparent}\n.brandbar{display:flex;align-items:center;gap:13px;padding:8px 4px 13px;border-bottom:1px solid #e5ece7;margin:0 0 9px;background:#fff}\n.brandmark{width:52px;height:52px;border-radius:50%;background:#e8f5e9;display:flex;align-items:center;justify-content:center;font-size:30px}\n.brandname{font-family:Manrope,sans-serif;font-weight:800;font-size:1.75rem;color:#123d2b;letter-spacing:-.04em}\n.brandtag{font-size:1rem;color:#53675d;border-left:1px solid #cbd8cf;padding-left:17px;margin-left:5px}\ndiv[role="radiogroup"]{display:flex;justify-content:flex-end;gap:10px;flex-wrap:wrap;margin-top:-3.45rem;margin-bottom:1.2rem}\ndiv[role="radiogroup"] label{background:#fff;border:0;border-radius:11px;padding:9px 15px;font-weight:700;color:#123d2b}\ndiv[role="radiogroup"] label:has(input:checked){background:#27864d;color:#fff}\n.hero-shell{display:grid;grid-template-columns:1fr 1fr;background:linear-gradient(100deg,#f4faf5,#eaf5ed);border-radius:0;overflow:hidden;margin:0 0 20px;min-height:350px}\n.hero-copy{min-height:350px;padding:25px 22px 22px 4px;display:flex;flex-direction:column;justify-content:center}\n.hero-chip{background:#e0f2e4;color:#155b35;padding:9px 15px;border-radius:28px;font-weight:700;font-size:.86rem;width:max-content;margin-bottom:17px}\n.hero-title{font-family:Manrope,sans-serif;font-size:clamp(2.1rem,3.5vw,3.25rem);line-height:1.04;letter-spacing:-.045em;color:#103a2b;font-weight:800;margin:0 0 17px}\n.hero-desc{font-size:1.08rem;line-height:1.55;color:#193e30;max-width:590px;margin:0 0 18px}\n.hero-shell [data-testid="stImage"]{height:100%;min-height:350px}\n.hero-shell [data-testid="stImage"] img{height:350px;object-fit:cover;border-radius:0}\n.stButton>button{border-radius:12px;font-weight:700;min-height:2.85rem;border:1px solid #dce9df}\n.stButton>button[kind="primary"]{background:#08713c;border-color:#08713c;color:#fff}\n.feature-card{background:#fff;border:1px solid #edf1ed;border-radius:20px;padding:18px 16px;min-height:122px;display:flex;align-items:center;gap:13px;box-shadow:0 6px 18px rgba(20,55,39,.035);margin:0 0 22px}\n.feature-icon{width:58px;height:58px;min-width:58px;border-radius:16px;background:#eaf6ed;color:#08763e;display:flex;align-items:center;justify-content:center;font-size:2rem;font-weight:700}\n.feature-copy b{font-size:.98rem;color:#103a2b}\n.feature-copy p{font-size:.84rem;line-height:1.45;color:#5d6b64;margin:5px 0 0}\n.class-panel{background:#fff;border:1px solid #dfeae2;border-radius:22px;padding:18px 20px 12px;margin:2px 0 12px;box-shadow:0 5px 18px rgba(20,55,39,.025)}\n.class-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;font-family:Manrope,sans-serif;font-weight:800;font-size:1.35rem;color:#103a2b;margin-bottom:14px}\n.class-guide-label{font-family:\'DM Sans\',sans-serif;font-size:.86rem;color:#155b35;background:#edf7ef;border-radius:24px;padding:9px 16px;font-weight:700}\n.class-label{background:#fff;border:1px solid #e1eae4;border-radius:0 0 12px 12px;text-align:center;min-height:48px;padding:7px 2px;font-size:.73rem;font-weight:700;color:#103a2b;line-height:1.2}\n.home-footnote{color:#607168;font-size:.81rem;padding:7px 2px 2px}\n[data-testid="stFileUploader"]{background:white;border:1.5px dashed #b8d2c1;border-radius:16px;padding:12px}\n[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px}\nfooter{visibility:hidden}#MainMenu{visibility:hidden}\n@media(max-width:1050px){.block-container{padding:0 1rem 2rem}div[role="radiogroup"]{margin-top:0;justify-content:flex-start}.hero-copy{min-height:260px;padding:20px 8px}.hero-title{font-size:2rem}.hero-shell [data-testid="stImage"] img{height:260px}.feature-card{min-height:130px}.brandtag{display:none}}\n</style>\n', unsafe_allow_html=True)

FRIENDLY = {
    "bacterial_leaf_blight": ("Bacterial leaf blight", "A bacterial disease that may cause water-soaked lesions and drying that spreads from leaf tips or edges.", "Look for yellowing or drying that begins near the leaf tip and progresses along the leaf."),
    "bacterial_leaf_streak": ("Bacterial leaf streak", "A bacterial disease associated with narrow, water-soaked streaks that may become translucent or brown.", "Inspect for narrow, elongated streaks. Similar symptoms can have other causes."),
    "bacterial_panicle_blight": ("Bacterial panicle blight", "A disease affecting the panicle and grain development. A leaf-only photograph may not show its key symptoms.", "Inspect the panicle and whole plant as well; a leaf image alone is insufficient for confirmation."),
    "blast": ("Rice blast", "A fungal disease that can produce spindle-shaped lesions with pale centres and darker margins.", "Check for spindle-shaped lesions and inspect other leaves and plant parts."),
    "brown_spot": ("Brown spot", "A fungal disease often associated with oval brown lesions, sometimes with a yellow halo.", "Look for oval brown spots, but confirm with local agricultural expertise."),
    "dead_heart": ("Dead heart", "A symptom in which the central shoot of a young rice tiller dries and may pull out easily.", "Inspect the central shoot and tiller; a single leaf photo may not be enough."),
    "downy_mildew": ("Downy mildew", "A disease that can cause pale or yellowish streaking and abnormal leaf growth.", "Check for pale streaking and distorted growth across multiple leaves."),
    "hispa": ("Rice hispa damage", "Feeding or scraping damage caused by the rice hispa beetle; leaves may show white streaks or scraped patches.", "Look for scraped leaf surfaces and white streaks; inspect for insect presence."),
    "normal": ("No disease pattern detected", "The model assigned this image to its normal-leaf class. This does not prove the plant is disease-free.", "Continue routine field monitoring. A healthy-looking leaf does not rule out other plant problems."),
    "tungro": ("Rice tungro", "A viral disease complex that may cause yellow-orange leaf discoloration and stunted plant growth.", "Check plant height, leaf colour and field-level spread; seek expert confirmation."),
}

TRANSFORM = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

@st.cache_resource(show_spinner="Loading the selected rice-leaf model…")
def load_model():
    model = build_model(len(CLASS_NAMES)).to(DEVICE)
    path = MODEL_DIR / CONFIG["primary_prediction_model"]
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def model_dropout_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.Dropout)]


def mc_dropout(model, x, passes=25):
    # Keep BatchNorm in eval mode; activate dropout layers only for stochastic passes.
    model.eval()
    drops = model_dropout_modules(model)
    for layer in drops:
        layer.train()
    outputs = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                outputs.append(torch.softmax(model(x), dim=1).cpu().numpy()[0])
    finally:
        model.eval()
    arr = np.asarray(outputs)
    mean_p = arr.mean(axis=0)
    std_p = arr.std(axis=0)
    entropy = float(-(mean_p * np.log(mean_p + 1e-12)).sum() / np.log(len(mean_p)))
    return mean_p, std_p, entropy, len(drops)


def make_gradcam(model, x, class_idx):
    activations, gradients = {}, {}
    target = model.features[-1]
    def forward_hook(_module, _inputs, output):
        activations["value"] = output
    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]
    h1 = target.register_forward_hook(forward_hook)
    h2 = target.register_full_backward_hook(backward_hook)
    try:
        model.eval(); model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
        acts, grads = activations["value"], gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
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
    return Image.blend(base, Image.fromarray(heat).resize(base.size), .38)


def friendly(key):
    return FRIENDLY.get(key, (key.replace("_", " ").title(), "Model-predicted category.", "Compare with field symptoms and seek expert advice."))


def section(title, desc=None):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if desc: st.markdown(f'<div class="muted">{desc}</div>', unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(f'<div class="panel"><div class="kicker">{label}</div><div class="big">{value}</div><div class="muted">{note}</div></div>', unsafe_allow_html=True)

# Shared brand header and four-item public navigation, matching the supplied reference.
st.markdown("""<div class="brandbar">
  <div class="brandmark">🌾</div>
  <div class="brandname">KrishiMitra</div>
  <div class="brandtag">Rice Leaf Disease Classification</div>
</div>""", unsafe_allow_html=True)

nav_options = ["🏠 Home", "▣ Detect Disease", "▤ Disease Guide", "ⓘ About"]
if "nav_target" in st.session_state:
    st.session_state["nav_choice"] = st.session_state.pop("nav_target")
selected_nav = st.radio("Main navigation", nav_options, key="nav_choice", horizontal=True, label_visibility="collapsed")
nav_to_page = {
    "🏠 Home": "🏠 Home",
    "▣ Detect Disease": "🌱 Detect disease",
    "▤ Disease Guide": "📚 Disease guide",
    "ⓘ About": "ℹ️ About and limitations",
}
page = nav_to_page[selected_nav]
st.sidebar.markdown("### KrishiMitra")
st.sidebar.caption("Rice leaf disease classification")
st.sidebar.markdown("---")
st.sidebar.markdown("**Research workspace**")
research_options = ["Not selected", "Research dashboard", "Experiments & methodology", "OOD evaluation"]
research_choice = st.sidebar.selectbox("Research tools", research_options, label_visibility="collapsed")
if research_choice == "Research dashboard":
    page = "📊 Research dashboard"
elif research_choice == "Experiments & methodology":
    page = "🧪 Experiments & methodology"
elif research_choice == "OOD evaluation":
    page = "🛡️ OOD evaluation"
st.sidebar.caption("Federated methods are research experiments, not public prediction settings.")

if page == "🏠 Home":
    hero_path = ROOT / "assets" / "rice_field_hero.jpg"
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
    hero_left, hero_right = st.columns([0.94, 1.06], gap="small")
    with hero_left:
        st.markdown("""
        <div class="hero-copy">
          <div class="hero-chip">▦ &nbsp; AI-Powered Education Tool</div>
          <h1 class="hero-title">Identify and Classify<br>Rice Leaf Diseases</h1>
          <p class="hero-desc">Upload a rice leaf image and get a disease classification, with a visual explanation and confidence result.</p>
        </div>
        """, unsafe_allow_html=True)
        action1, action2 = st.columns([1.2, .8], gap="small")
        with action1:
            if st.button("Detect Disease Now  →", type="primary", use_container_width=True):
                st.session_state["nav_target"] = "▣ Detect Disease"
                st.rerun()
        with action2:
            if st.button("ⓘ  Learn More", use_container_width=True):
                st.session_state["nav_target"] = "ⓘ About"
                st.rerun()
    with hero_right:
        if hero_path.exists():
            st.image(Image.open(hero_path), use_container_width=True)
        else:
            st.markdown('<div class="hero-fallback">Rice field</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    feature_data = [
        ("◎", "Accurate Classification", "Identify one of 10 rice leaf categories."),
        ("◇", "Lightweight & Efficient", "Fast and resource-friendly model (MobileNetV3)."),
        ("◉", "Explainable Result", "See important leaf areas with Grad-CAM."),
        ("⬡", "Reliable Prediction", "View prediction confidence and uncertainty.")
    ]
    feature_cols = st.columns(4, gap="medium")
    for col, (symbol, title, description) in zip(feature_cols, feature_data):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{symbol}</div><div class="feature-copy"><b>{title}</b><p>{description}</p></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="class-panel"><div class="class-heading"><span>🍃 &nbsp; 10 Supported Classes</span><span class="class-guide-label">View Disease Guide &nbsp; →</span></div>', unsafe_allow_html=True)
    class_cols = st.columns(10, gap="small")
    thumb_dir = ROOT / "assets" / "disease_thumbnails"
    for col, key in zip(class_cols, CLASS_NAMES):
        with col:
            thumb = thumb_dir / f"{key}.jpg"
            if thumb.exists():
                st.image(Image.open(thumb), use_container_width=True)
            st.markdown(f'<div class="class-label">{friendly(key)[0]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-footnote">For best results, upload a clear rice leaf photograph taken in natural light. The model provides a research decision-support prediction, not a guaranteed diagnosis.</div>', unsafe_allow_html=True)

elif page == "📚 Disease guide":
    section("Rice disease guide", "Learn about the ten categories supported by the KrishiMitra classification model.")
    cols = st.columns(2, gap="medium")
    for idx, key in enumerate(CLASS_NAMES):
        title, description, guidance = friendly(key)
        with cols[idx % 2]:
            st.markdown(f'<div class="panel" style="margin-bottom:12px"><b>{title}</b><div class="muted" style="margin-top:7px">{description}</div><div class="muted" style="margin-top:8px"><b>What to check:</b> {guidance}</div></div>', unsafe_allow_html=True)

elif page == "🌱 Detect disease":
    section("Analyse a rice-leaf image", "Upload a clear photograph. The model returns its most likely supported class and an explanation—not a guaranteed diagnosis.")
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        uploaded = st.file_uploader("Choose a leaf photograph", type=["jpg", "jpeg", "png", "webp"], help="Use a clear, original photograph of one rice leaf.")
    with right:
        st.markdown('<div class="panel"><b>📷 Photo tips</b><div class="muted" style="margin-top:8px">• Photograph one leaf in natural light<br>• Keep the affected area in focus<br>• Avoid shadows, blur and filters<br>• Include the full leaf where possible</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="warn"><b>Research status:</b> Quantitative OOD validation using PlantVillage is still pending in the recorded research. This release therefore does not automatically certify that an uploaded image is a rice leaf or reliably reject every unfamiliar image. Use rice-leaf photographs only.</div>', unsafe_allow_html=True)
    if uploaded:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            if min(image.size) < 100:
                st.error("The image is too small. Please upload a clearer photograph with a longer side of at least 100 pixels.")
            else:
                # Conservative quality heuristics; they are not a trained image-quality model.
                gray = np.asarray(image.convert("L"), dtype=np.float32)
                contrast = float(gray.std())
                if contrast < 12:
                    st.warning("This image has very low contrast. Consider taking another photo in natural light.")
                st.image(image, caption="Uploaded image", width=360)
                if st.button("Analyse image", type="primary", use_container_width=True):
                    with st.spinner("Running model inference and explanation…"):
                        model = load_model()
                        x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
                        probs, variability, entropy, dropout_count = mc_dropout(model, x, 25)
                        pred_idx = int(np.argmax(probs)); pred_key = CLASS_NAMES[pred_idx]
                        title, description, guidance = friendly(pred_key)
                        cam = make_gradcam(model, x, pred_idx)
                        st.session_state["prediction_result"] = {
                            "image": image, "probs": probs, "variability": variability,
                            "entropy": entropy, "dropout_count": dropout_count,
                            "pred_idx": pred_idx, "pred_key": pred_key, "title": title,
                            "description": description, "guidance": guidance, "cam": cam,
                        }
                result = st.session_state.get("prediction_result")
                if result is not None:
                    st.markdown("---")
                    section("Prediction result", "The highest-scoring class from the selected model.")
                    a, b = st.columns([1, 1], gap="large")
                    with a:
                        st.markdown(f'<div class="soft"><div class="kicker">Most likely model class</div><div class="big">{result["title"]}</div><div class="muted">{result["description"]}</div></div>', unsafe_allow_html=True)
                        st.metric("Model probability", f'{result["probs"][result["pred_idx"]]:.1%}')
                        st.caption("This is the model's mean softmax score across 25 passes. It is not a guarantee of diagnostic correctness or a validated calibrated confidence for this checkpoint.")
                    with b:
                        st.markdown('<div class="panel"><b>Reliability indicators</b>', unsafe_allow_html=True)
                        st.metric("Normalized predictive entropy", f'{result["entropy"]:.3f}', help="0 is more concentrated; 1 is maximally spread across the 10 classes.")
                        st.metric("Top-class variation across passes", f'{result["variability"][result["pred_idx"]]:.4f}')
                        if result["dropout_count"] == 0:
                            st.warning("No dropout layers were found. Repeated passes may be identical; MC Dropout uncertainty is not available for this checkpoint architecture.")
                        else:
                            st.info("These are exploratory uncertainty indicators. No validated reject threshold is configured, so they should not be interpreted as a certified reliable/uncertain decision.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    section("Visual explanation", "Grad-CAM highlights image regions that influenced the selected class score.")
                    c1, c2 = st.columns(2)
                    with c1: st.image(result["image"], caption="Original image", use_container_width=True)
                    with c2: st.image(overlay_cam(result["image"], result["cam"]), caption="Grad-CAM overlay", use_container_width=True)
                    st.markdown(f'<div class="info"><b>What this may mean:</b> {result["guidance"]}<br><br>Grad-CAM is an explanation aid, not lesion segmentation or proof that the highlighted area is the biological cause of disease.</div>', unsafe_allow_html=True)
                    with st.expander("View alternative model classes"):
                        order = np.argsort(result["probs"])[::-1]
                        for i in order[:5]:
                            nm = friendly(CLASS_NAMES[int(i)])[0]
                            st.write(f"**{nm}** · {result['probs'][i]:.1%}")
                            st.progress(float(result["probs"][i]))
                    st.markdown('<div class="warn"><b>Next step:</b> Compare the result with visible field symptoms. For treatment decisions, confirm with a local agriculture officer or Krishi Vigyan Kendra (KVK). Do not select pesticides solely from this model output.</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error("The image could not be analysed. Try a clear JPG, PNG or WEBP photograph.")
            with st.expander("Technical diagnostic details"): st.code(str(exc))
    else:
        st.markdown('<div class="panel" style="text-align:center;padding:30px"><div style="font-size:2.1rem">🌿</div><b>Your analysis will appear here</b><div class="muted">Upload a rice-leaf photograph to begin.</div></div>', unsafe_allow_html=True)

elif page == "📊 Research dashboard":
    section("Research overview", "A consolidated view of the recorded KrishiMitra research experiments. Results from different report snapshots are kept separate where evaluation protocols differ.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Labelled images", "10,407", "Recorded rice-leaf dataset")
    with c2: metric_card("Disease classes", "10", "Supported classification labels")
    with c3: metric_card("Federated clients", "5", "Simulated client partitions")
    with c4: metric_card("Backbone", "MobileNetV3-S", "Small architecture")
    section("Centralized baseline", "Master research report snapshot: held-out internal test split.")
    m1,m2,m3 = st.columns(3)
    with m1: metric_card("Test accuracy", "95.97%", "Centralized MobileNetV3-Small")
    with m2: metric_card("Test Macro-F1", "95.69%", "Equal-weighted class F1 average")
    with m3: metric_card("Test samples", "1,562", "Held-out labelled images")
    section("Federated strategy comparison", "Consolidated master-report comparison. FedAvg/FedProx are global-model results; FedBN/FedPer are means across five client-specific models.")
    df = pd.DataFrame({"Method":["FedAvg","FedProx","FedBN","FedPer"],"IID Macro-F1 (%)":[62.71,63.22,62.69,61.72],"Non-IID Macro-F1 (%)":[55.97,56.79,55.15,37.18]})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Method"), use_container_width=True)
    st.markdown('<div class="info"><b>Interpretation:</b> FedProx recorded the highest Macro-F1 among the federated strategies in both settings in this consolidated comparison. The centralized baseline has a different training/evaluation setup and must not be treated as a like-for-like federated comparison.</div>', unsafe_allow_html=True)
    section("Additional calibration experiment", "Separate recorded experiment: temperature T=0.7751, ECE 0.0896 → 0.0347, Brier 0.3159 → 0.3034. Do not apply these values to the deployed checkpoint unless its matching calibration artifact is verified.")
    q1,q2,q3 = st.columns(3)
    with q1: metric_card("Temperature", "0.7751", "Validation-fitted value in separate experiment")
    with q2: metric_card("ECE", "0.0896 → 0.0347", "Reported before and after scaling")
    with q3: metric_card("Brier score", "0.3159 → 0.3034", "Reported before and after scaling")
    section("Research completion status")
    status = pd.DataFrame([
        ["Dataset preparation and split","Completed"],["Centralized MobileNetV3-Small","Completed"],["IID / Non-IID partitioning","Completed"],
        ["FedAvg / FedProx experiments","Completed"],["Personalized FL initial experiments","Completed"],["Grad-CAM analysis","Completed"],
        ["Temperature scaling experiment","Completed in separate snapshot"],["PlantVillage OOD evaluation","In progress / pending validation"],
        ["Quantitative uncertainty validation","Not established as completed"],["External field validation","Pending"]
    ],columns=["Research component","Recorded status"])
    st.dataframe(status,use_container_width=True,hide_index=True)

elif page == "🧪 Experiments & methodology":
    section("Dataset and preprocessing", "Recorded setup from the research progress report.")
    d1,d2,d3 = st.columns(3)
    with d1: metric_card("Training", "7,284", "70% of 10,407 images")
    with d2: metric_card("Validation", "1,561", "15% model selection")
    with d3: metric_card("Internal test", "1,562", "15% held-out evaluation")
    st.markdown('<div class="panel"><b>Preprocessing pipeline</b><div class="muted">Image integrity inspection → stratified split (seed 42) → resize to 224 × 224 → tensor conversion → ImageNet normalization → class-aware training setup.</div></div>',unsafe_allow_html=True)
    st.write("**Ten supported classes**")
    class_titles = [friendly(x)[0] for x in CLASS_NAMES]
    st.write(" · ".join(class_titles))
    section("Model training configuration")
    cfg = pd.DataFrame([["Architecture","ImageNet-pretrained MobileNetV3-Small"],["Output classes","10"],["Centralized epochs","20"],["Optimizer","AdamW"],["Learning rate","0.0001"],["Weight decay","0.0001"],["Input resolution","224 × 224"],["Federated clients","5"],["Non-IID partition","Dirichlet α = 0.5 (recorded setup)"],["FedProx coefficient","µ = 0.01 (recorded setup in progress report)"],["Federated schedule","See experiment-specific run; do not merge differing round counts"]],columns=["Setting","Recorded configuration"])
    st.dataframe(cfg,use_container_width=True,hide_index=True)
    section("Federated learning workflow")
    st.markdown('<div class="soft"><b>Client-side training</b> → clients train on local partitions → share model updates → server aggregates shared parameters → updated model returns to clients. Raw images remain in their local partitions in this simulation.</div>',unsafe_allow_html=True)
    st.markdown("- **FedAvg:** sample-weighted averaging of client model parameters.\n- **FedProx:** adds a proximal regularization term to limit local drift from the global model.\n- **FedBN:** keeps Batch Normalization state local to adapt to client-specific feature distributions.\n- **FedPer:** shares the feature backbone while retaining client-specific classifier heads.")
    st.markdown('<div class="warn"><b>Implementation scope:</b> Your reports describe single-process federated simulations, not deployment across physical farms/devices. Secure aggregation and differential privacy were not reported as implemented. Federated learning by itself should not be described as a formal privacy guarantee.</div>',unsafe_allow_html=True)
    
    section("Research problem-to-solution mapping")
    mapping = pd.DataFrame([
        ["Manual / difficult disease identification", "10-class rice-leaf image classification"],
        ["Privacy concerns from central image collection", "Federated training experiments; raw images stay in simulated client partitions"],
        ["Heterogeneous farm data (Non-IID)", "IID/Non-IID client partitioning and FedAvg/FedProx/FedBN/FedPer comparison"],
        ["Limited interpretability", "Grad-CAM heatmap for selected class predictions"],
        ["Overconfident or unfamiliar inputs", "Calibration and planned OOD/uncertainty validation"],
        ["Low-resource deployment", "MobileNetV3-Small lightweight architecture"],
        ["Usability for non-technical users", "Plain-language prediction, symptom context and responsible next steps"],
    ], columns=["Research problem", "System component / experiment"])
    st.dataframe(mapping, use_container_width=True, hide_index=True)

    section("Explainability, uncertainty and OOD")
    st.markdown("**Grad-CAM:** completed for selected examples using the trained centralized checkpoint. It highlights regions influencing the class score; it is not causal proof or lesion segmentation.\n\n**Uncertainty and calibration:** temperature scaling was reported in a separate experiment. MC Dropout and predictive entropy should be validated for the final deployed checkpoint before being used as rejection rules.\n\n**OOD / PlantVillage:** PlantVillage is an external dataset for testing unfamiliar/non-rice images. It is not an eleventh disease class and should not be mixed into rice disease training labels. Quantitative OOD evaluation remains pending in the available project record.")


elif page == "🛡️ OOD evaluation":
    section("Out-of-distribution evaluation workspace",
            "Use this workspace to evaluate whether a scoring method separates supported rice-leaf images from external unfamiliar images. PlantVillage is external OOD data—not an additional rice disease class.")
    st.markdown('<div class="warn"><b>Current research status:</b> Your recorded reports mark quantitative OOD evaluation as pending. No validated OOD threshold is bundled with this application. This workspace calculates metrics only from score files that you provide; it does not pretend OOD rejection is already validated.</div>', unsafe_allow_html=True)

    st.markdown("### 1. Prepare a score file")
    st.write("Upload a CSV containing one row per image and these columns:")
    st.code("split,is_ood,msp,energy\nvalidation,0,0.91,-8.2\nvalidation,1,0.42,-2.1\ntest,0,0.88,-7.4\ntest,1,0.35,-1.8", language="csv")
    st.caption("is_ood: 0 = in-distribution rice image, 1 = out-of-distribution image. MSP is maximum softmax probability. Energy must use one consistent definition and sign convention for every row.")
    template = pd.DataFrame([
        ["validation", 0, 0.91, -8.2],
        ["validation", 1, 0.42, -2.1],
        ["test", 0, 0.88, -7.4],
        ["test", 1, 0.35, -1.8],
    ], columns=["split", "is_ood", "msp", "energy"])
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="krishimitra_ood_scores_template.csv", mime="text/csv")
    uploaded_scores = st.file_uploader("Upload your real OOD score CSV", type=["csv"], key="ood_scores_csv")
    if uploaded_scores is not None:
        try:
            scores = pd.read_csv(uploaded_scores)
            required = {"split", "is_ood", "msp", "energy"}
            missing = required - set(scores.columns)
            if missing:
                st.error("Missing required columns: " + ", ".join(sorted(missing)))
            elif not set(scores["is_ood"].dropna().astype(int).unique()).issubset({0, 1}):
                st.error("is_ood must contain only 0 and 1.")
            elif not {"validation", "test"}.issubset(set(scores["split"].astype(str).str.lower())):
                st.error("Include separate validation and test rows. Do not tune thresholds on the test split.")
            else:
                scores["split"] = scores["split"].astype(str).str.lower()
                st.success(f"Loaded {len(scores):,} score records.")
                from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
                from sklearn.metrics import confusion_matrix
                def score_metrics(frame, col, ood_high=True):
                    y = frame["is_ood"].astype(int).to_numpy()
                    raw = pd.to_numeric(frame[col], errors="coerce").to_numpy()
                    valid = np.isfinite(raw)
                    y, raw = y[valid], raw[valid]
                    if len(np.unique(y)) < 2:
                        raise ValueError("Each split must contain both ID and OOD examples.")
                    s = raw if ood_high else -raw
                    auc = roc_auc_score(y, s)
                    aupr = average_precision_score(y, s)
                    fpr, tpr, thresholds = roc_curve(y, s)
                    idx = int(np.argmin(np.abs(tpr - 0.95)))
                    return auc, aupr, float(fpr[idx]), float(thresholds[idx]), y, s
                val = scores[scores["split"] == "validation"]
                test = scores[scores["split"] == "test"]
                st.markdown("### 2. Validation-set threshold selection")
                method = st.selectbox("OOD score", ["MSP (lower means more OOD)", "Energy (higher means more OOD)"])
                col = "msp" if method.startswith("MSP") else "energy"
                high = method.startswith("Energy")
                aucv, auprv, fpr95v, thr, yv, sv = score_metrics(val, col, ood_high=high)
                st.metric("Validation AUROC", f"{aucv:.3f}")
                st.metric("Validation AUPR", f"{auprv:.3f}")
                st.metric("Validation FPR at approximately 95% TPR", f"{fpr95v:.3f}")
                st.write(f"Validation-derived candidate threshold: **{thr:.6g}**")
                st.caption("This threshold is selected from validation scores only. Confirm its behaviour on the separate test split before deployment.")
                st.markdown("### 3. Held-out test evaluation")
                auct, auprt, fpr95t, _, yt, stest = score_metrics(test, col, ood_high=high)
                m1, m2, m3 = st.columns(3)
                m1.metric("Test AUROC", f"{auct:.3f}")
                m2.metric("Test AUPR", f"{auprt:.3f}")
                m3.metric("Test FPR near 95% TPR", f"{fpr95t:.3f}")
                # Thresholded test confusion matrix, using only validation-selected threshold.
                pred = (stest >= thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0,1]).ravel()
                st.write("Test results at the validation-selected threshold")
                st.dataframe(pd.DataFrame([
                    ["ID correctly accepted", int(tn)], ["ID incorrectly rejected", int(fp)],
                    ["OOD incorrectly accepted", int(fn)], ["OOD correctly flagged", int(tp)]
                ], columns=["Outcome", "Images"]), use_container_width=True, hide_index=True)
                st.warning("These metrics are valid only if your CSV contains genuine, correctly labelled, independent validation and test scores generated by the intended deployed checkpoint. Do not use the illustrative template rows as experimental evidence.")
        except Exception as exc:
            st.error(f"Could not evaluate the supplied scores: {exc}")


else:
    section("About KrishiMitra")
    st.markdown("KrishiMitra is the public application built from the FedRice-XAI research, which investigates lightweight rice-leaf disease classification, federated optimization under heterogeneous client data, personalization, explainability and prediction reliability. The public application uses one selected centralized MobileNetV3-Small checkpoint. Federated strategies are presented as research experiments rather than user-facing model choices.")
    section("What this application can and cannot claim")
    st.markdown("**Implemented in the supplied application:** centralized 10-class inference, repeated-pass score variability where dropout layers exist, Grad-CAM visualization, research experiment dashboard and methodology summary.\n\n**Not yet validated for real-world diagnosis:** automatic rice/non-rice rejection, calibrated uncertainty rejection thresholds for this deployed checkpoint, PlantVillage OOD detection metrics, external field performance and treatment recommendations.")
    st.markdown('<div class="warn"><b>Responsible-use notice:</b> This is a research decision-support prototype, not a certified plant diagnostic service. Image classification may be wrong, especially for unfamiliar varieties, poor lighting, mixed symptoms or images outside the training distribution. Confirm important decisions with agricultural experts.</div>',unsafe_allow_html=True)
    section("Research limitations")
    st.markdown("- Recorded results use one dataset split and limited random-seed evidence.\n- Centralized and federated compute budgets differ.\n- FedBN/FedPer scores are client means, unlike global FedAvg/FedProx scores.\n- Grad-CAM examples are selected and do not establish general explanation quality.\n- OOD and external field validation remain open work.\n- No combined ensemble accuracy has been reported.")

st.markdown("<hr style='border:0;border-top:1px solid #e2eae4;margin:2rem 0 1rem'>",unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#718078;font-size:.82rem">KrishiMitra · Research prototype · MobileNetV3-Small · Decision support, not a substitute for agricultural diagnosis</div>',unsafe_allow_html=True)
