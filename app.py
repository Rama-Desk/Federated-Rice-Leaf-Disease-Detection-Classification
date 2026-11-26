from pathlib import Path
import json
import hashlib

import numpy as np
import streamlit as st
st.set_page_config(page_title="KrishiMitra | Dual Model Rice Disease Support", page_icon="🌾", layout="wide")
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model import build_model

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

DATASETS = {
    "Dataset 1": {
        "checkpoint": "dataset1_centralized_best.pth",
        "classes_file": "dataset1_classes.json",
        "description": "Rice disease dataset with Healthy, Leaf Blast, Leaf Scald, Neck Blast and other classes.",
        "metric_note": "Checkpoint validation snapshot: 94.34% accuracy and 94.12% macro-F1 (validation set).",
        "checkpoint_format": "wrapped",
    },
    "Dataset 2": {
        "checkpoint": "dataset2_centralized_best.pth",
        "classes_file": "dataset2_classes.json",
        "description": "Paddy disease classification dataset with ten classes including Bacterial Leaf Streak, Dead Heart and Downy Mildew.",
        "metric_note": "Research report test snapshot: 95.97% accuracy and 95.69% macro-F1 (1,562 held-out images).",
        "checkpoint_format": "state_dict",
    },
}

FRIENDLY = {
    "healthy": ("Healthy leaf", "The model places this image in its healthy-leaf category. This does not guarantee that the whole plant is disease-free."),
    "normal": ("Normal leaf", "The model places this image in its normal category. Continue checking other leaves and the whole plant."),
    "bacterial_leaf_blight": ("Bacterial leaf blight", "A bacterial disease often associated with water-soaked lesions and yellowing or drying that can progress from leaf tips or margins."),
    "bacterial_leaf_streak": ("Bacterial leaf streak", "Often associated with narrow, elongated, water-soaked streaks that may become brown."),
    "bacterial_panicle_blight": ("Bacterial panicle blight", "This disease mainly affects panicles and grain development; a leaf-only photograph may not show its key signs."),
    "blast": ("Rice blast", "A fungal disease that can produce spindle-shaped lesions with pale centres and darker margins."),
    "leaf_blast": ("Leaf blast", "A blast symptom on leaves, often showing spindle-shaped lesions with pale centres and darker edges."),
    "neck_blast": ("Neck blast", "A blast form affecting the neck of the panicle. Inspect the panicle neck and grain-bearing parts of the plant."),
    "brown_spot": ("Brown spot", "Can produce oval or circular brown lesions, sometimes with yellow halos."),
    "leaf_scald": ("Leaf scald", "Can cause lesions that begin near leaf tips or edges and develop into larger scalded areas."),
    "narrow_brown_spot": ("Narrow brown spot", "Often appears as narrow, elongated brown lesions on leaves."),
    "dead_heart": ("Dead heart", "The central shoot of a young rice tiller dries and may pull out easily. Inspect the whole tiller."),
    "downy_mildew": ("Downy mildew", "May cause pale yellow or whitish streaks and abnormal leaf growth."),
    "hispa": ("Rice hispa damage", "Feeding or scraping by rice hispa beetles may leave white streaks or scraped patches on leaves."),
    "rice_hispa": ("Rice hispa damage", "Feeding or scraping by rice hispa beetles may leave white streaks or scraped patches on leaves."),
    "sheath_blight": ("Sheath blight", "A fungal disease that commonly produces lesions on leaf sheaths and may spread to leaf blades."),
    "tungro": ("Rice tungro", "A viral disease complex that may cause yellow-orange leaf discoloration and stunted plant growth."),
}


def pretty_name(class_name: str) -> str:
    return FRIENDLY.get(class_name.lower(), (class_name.replace("_", " ").title(), ""))[0]


def class_description(class_name: str) -> str:
    return FRIENDLY.get(class_name.lower(), ("", "No description available."))[1]


@st.cache_resource(show_spinner="Loading both centralized models…")
def load_all_models():
    loaded = {}
    for dataset_name, cfg in DATASETS.items():
        class_names = json.loads((MODEL_DIR / cfg["classes_file"]).read_text(encoding="utf-8"))
        checkpoint_path = MODEL_DIR / cfg["checkpoint"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path.name}")
        model = build_model(len(class_names)).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        if cfg["checkpoint_format"] == "wrapped":
            state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
            stored_classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
            if not state:
                raise ValueError(f"{dataset_name}: model_state_dict not found in checkpoint.")
            if stored_classes and list(stored_classes) != class_names:
                raise ValueError(f"{dataset_name}: class order in checkpoint does not match class JSON.")
        else:
            state = checkpoint
        model.load_state_dict(state, strict=True)
        model.eval()
        loaded[dataset_name] = {"model": model, "classes": class_names, "config": cfg}
    return loaded


def gradcam(model, input_tensor, class_index):
    captured = {}
    target_layer = model.features[-1]

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor.detach().clone().requires_grad_(True))
            logits[0, class_index].backward()
        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM activation or gradient was not captured.")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        low, high = float(cam.min()), float(cam.max())
        if high - low < 1e-8:
            raise RuntimeError("Grad-CAM map is almost uniform for this image.")
        return np.clip((cam - low) / (high - low), 0, 1)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.eval()


def make_overlay(image, heatmap):
    base = np.asarray(image.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB"), dtype=np.float32)
    h = np.clip(heatmap, 0, 1)
    # Jet-like colour map generated with NumPy; no plotting dependency required.
    red = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    colour = np.stack([red, green, blue], axis=-1) * 255
    return Image.fromarray(np.uint8(np.clip(0.56 * base + 0.44 * colour, 0, 255)))


def run_prediction(model_info, image, passes=25):
    """Return class scores, exploratory MC-dropout uncertainty and OOD screening scores."""
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    model = model_info["model"]
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    samples = []
    model.eval()  # Keep BatchNorm layers fixed during stochastic passes.
    try:
        if dropout_layers:
            for layer in dropout_layers:
                layer.train()
            with torch.no_grad():
                for _ in range(passes):
                    samples.append(torch.softmax(model(tensor), dim=1)[0].cpu().numpy())
        else:
            with torch.no_grad():
                samples.append(torch.softmax(model(tensor), dim=1)[0].cpu().numpy())
    finally:
        model.eval()

    sample_array = np.asarray(samples, dtype=np.float64)
    mean_scores = sample_array.mean(axis=0)
    std_scores = sample_array.std(axis=0)
    ranking = np.argsort(mean_scores)[::-1]
    best_index = int(ranking[0])
    eps = 1e-12
    predictive_entropy = float(-(mean_scores * np.log(mean_scores + eps)).sum() / np.log(len(mean_scores)))
    sample_entropies = -(sample_array * np.log(sample_array + eps)).sum(axis=1) / np.log(len(mean_scores))
    expected_entropy = float(sample_entropies.mean())
    mutual_information = float(max(0.0, predictive_entropy - expected_entropy))
    # Energy score is reported as a raw research diagnostic, not a calibrated OOD threshold.
    with torch.no_grad():
        deterministic_logits = model(tensor)[0].detach().cpu().numpy().astype(np.float64)
    max_softmax = float(mean_scores[best_index])
    energy_score = float(-np.log(np.exp(deterministic_logits - deterministic_logits.max()).sum()) - deterministic_logits.max())

    cam = None
    cam_error = None
    try:
        cam = gradcam(model, tensor, best_index)
    except Exception as exc:
        cam_error = str(exc)
    finally:
        model.eval()
    return {
        "scores": mean_scores, "ranking": ranking, "best_index": best_index,
        "cam": cam, "cam_error": cam_error, "std_scores": std_scores,
        "predictive_entropy": predictive_entropy, "expected_entropy": expected_entropy,
        "mutual_information": mutual_information, "dropout_layers": len(dropout_layers),
        "mc_passes": len(samples), "max_softmax": max_softmax, "energy_score": energy_score,
    }


def canonical_class(name):
    key = name.lower()
    aliases = {
        "healthy": "normal", "normal": "normal", "rice_hispa": "hispa",
        "leaf_blast": "blast", "neck_blast": "blast",
    }
    return aliases.get(key, key)


def show_model_result(dataset_name, info, result, image):
    classes = info["classes"]
    scores = result["scores"]
    order = result["ranking"]
    best = int(result["best_index"])
    best_class = classes[best]
    best_score = float(scores[best])
    second_score = float(scores[int(order[1])])
    margin = best_score - second_score

    st.markdown(f"### {dataset_name} — {pretty_name(best_class)}")
    st.caption(info["config"]["description"])
    st.metric("Highest model score", f"{best_score:.1%}", help="Softmax score across this model's own classes. It is not a calibrated disease probability.")
    st.markdown(f"**What this class means:** {class_description(best_class)}")
    if best_score < 0.50 or margin < 0.10:
        st.warning(f"Low separation between leading classes (top-two margin: {margin:.1%}). Treat this prediction cautiously.")
    else:
        st.info(f"The leading class is ahead of the second class by {margin:.1%}. This is model preference, not proof of diagnosis.")

    st.markdown("**Uncertainty and OOD screening indicators**")
    u1, u2 = st.columns(2)
    u1.metric("Normalized predictive entropy", f"{result['predictive_entropy']:.3f}", help="0 means the scores are concentrated; values nearer 1 mean the model spreads its scores across classes.")
    u2.metric("Top-class variation", f"{float(result['std_scores'][best]):.4f}", help="Standard deviation across repeated MC-dropout passes. This is exploratory uncertainty, not a calibrated error probability.")
    u3, u4 = st.columns(2)
    u3.metric("Mutual information", f"{result['mutual_information']:.3f}", help="An exploratory estimate of model uncertainty from variation across stochastic passes.")
    u4.metric("Energy score", f"{result['energy_score']:.3f}", help="Raw energy-based OOD research score. It has no validated threshold in this project.")
    if result["dropout_layers"] == 0:
        st.warning("This checkpoint has no dropout layers. MC-dropout uncertainty is unavailable; only the single-pass score distribution is shown.")
    elif result["predictive_entropy"] >= 0.65 or result["max_softmax"] < 0.50:
        st.warning("Reliability screening flag: the score distribution is diffuse or the top score is low. The image may be ambiguous or unfamiliar. Retake the photo and seek expert confirmation.")
    else:
        st.info("No heuristic screening flag was raised. This is not proof that the image is in-distribution or that the prediction is correct.")
    st.caption(f"MC Dropout: {result['mc_passes']} forward passes. OOD screening is exploratory; no validated rejection threshold is configured.")

    st.markdown("**Top five classes considered**")
    for rank, idx in enumerate(order[:5], start=1):
        idx = int(idx)
        st.write(f"{rank}. {pretty_name(classes[idx])} — {scores[idx]:.1%}")
        st.progress(float(scores[idx]))

    st.markdown("**Grad-CAM visual explanation**")
    if result["cam"] is not None:
        st.image(make_overlay(image, result["cam"]), caption=f"{dataset_name}: regions influencing {pretty_name(best_class)} score", use_container_width=True)
        st.caption("Warm colours show areas with greater influence on this class score. This is not a disease-lesion detector and does not prove the model focused on the correct symptom.")
    else:
        st.warning("Grad-CAM could not be generated for this model and image. The classification output is still available.")
        if result["cam_error"]:
            st.caption(result["cam_error"])
    st.caption(info["config"]["metric_note"])


def render_home():
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("<div class='hero'><span class='eyebrow'>AI-assisted crop support</span><h1>Understand rice leaf disease predictions</h1><p>Upload one clear rice-leaf photograph. Two independently trained centralized models analyse it, show their leading classes, compare alternatives and provide Grad-CAM visual explanations.</p></div>", unsafe_allow_html=True)
        if st.button("Start disease detection →", type="primary", use_container_width=True):
            st.session_state["nav"] = "Detect Disease"
            st.rerun()
    with right:
        hero = ASSET_DIR / "rice_field_hero.jpg"
        if hero.exists():
            st.image(Image.open(hero), use_container_width=True)
    st.markdown("### What KrishiMitra provides")
    a, b, c, d = st.columns(4)
    a.markdown("<div class='feature'><b>Dual-model classification</b><p>Two centralized models independently analyse the same image.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='feature'><b>Explainable prediction</b><p>Top-five classes, score comparisons and disease descriptions.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='feature'><b>Grad-CAM visualisation</b><p>Highlights image regions influencing each model's selected class.</p></div>", unsafe_allow_html=True)
    d.markdown("<div class='feature'><b>Uncertainty and OOD checks</b><p>Entropy, MC-dropout variation, mutual information and energy score.</p></div>", unsafe_allow_html=True)
    st.info("The two datasets use different disease labels. Their raw scores must not be averaged or treated as directly comparable probabilities.")


def render_detection():
    st.markdown("## Detect and compare")
    st.write("Upload a clear, unfiltered rice-leaf image. Both centralized models will analyse the same image using their own class lists.")
    upload = st.file_uploader("Choose a rice-leaf image", type=["jpg", "jpeg", "png", "webp"], help="Use a focused photograph in natural light. Avoid screenshots, collages and heavily edited images.")
    if not upload:
        st.markdown("<div class='empty'><b>Ready for image analysis</b><p>Upload an image to receive two independent classification results.</p></div>", unsafe_allow_html=True)
        return
    token = hashlib.sha256(upload.getvalue()).hexdigest()
    try:
        image = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
    except Exception:
        st.error("This image file could not be opened. Try a standard JPG or PNG photograph.")
        return
    if min(image.size) < 120:
        st.error("The image is too small. Please upload a clearer image with at least 120 pixels on each side.")
        return
    st.image(image, caption="Uploaded rice-leaf image", width=460)
    if st.button("Run both models", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading both models and generating explanations…"):
                models = load_all_models()
                predictions = {name: run_prediction(info, image) for name, info in models.items()}
            st.session_state["dual_result"] = {"token": token, "image": image, "models": models, "predictions": predictions}
        except Exception as exc:
            st.error("The models could not complete prediction. Check that both checkpoint files are present and compatible.")
            with st.expander("Technical error details"):
                st.code(str(exc))
            return

    result = st.session_state.get("dual_result")
    if not result or result.get("token") != token:
        return
    models = result["models"]
    predictions = result["predictions"]
    st.markdown("---")
    st.markdown("## Classification results")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            show_model_result("Dataset 1 model", models["Dataset 1"], predictions["Dataset 1"], image)
    with col2:
        with st.container(border=True):
            show_model_result("Dataset 2 model", models["Dataset 2"], predictions["Dataset 2"], image)

    c1 = models["Dataset 1"]["classes"][predictions["Dataset 1"]["best_index"]]
    c2 = models["Dataset 2"]["classes"][predictions["Dataset 2"]["best_index"]]
    k1, k2 = canonical_class(c1), canonical_class(c2)
    st.markdown("## Model comparison")
    if k1 == k2:
        st.success(f"Both models selected the same broad category: {pretty_name(k1)}. This is agreement between two model outputs, not independent confirmation of a field diagnosis.")
    else:
        st.warning("The two models selected different categories. This is a model disagreement; do not choose a disease or treatment based only on the higher displayed score.")
        st.write(f"- Dataset 1: **{pretty_name(c1)}**")
        st.write(f"- Dataset 2: **{pretty_name(c2)}**")
        st.write("The models were trained with different datasets and class definitions, so their scores are not directly comparable.")
    st.markdown("### Overall reliability interpretation")
    entropies = [predictions[n]["predictive_entropy"] for n in ("Dataset 1", "Dataset 2")]
    variations = [float(predictions[n]["std_scores"][predictions[n]["best_index"]]) for n in ("Dataset 1", "Dataset 2")]
    if max(entropies) >= 0.65 or max(variations) >= 0.08:
        st.warning("At least one model shows elevated exploratory uncertainty. Treat the classification as tentative and verify the visible symptoms.")
    else:
        st.info("Neither model triggered the simple uncertainty review flag. This does not establish that the image is familiar to the models or that the diagnosis is correct.")
    st.markdown("**OOD limitation:** Maximum softmax, normalized entropy and energy are displayed as research screening signals. The project record does not contain a validated rice-versus-external-image threshold or completed PlantVillage OOD metrics. KrishiMitra therefore does not claim automatic non-rice rejection.")
    with st.expander("How to interpret the two-model comparison and reliability scores"):

        st.write("Each model produces a score distribution over its own ten classes. The app shows each model's top class and alternatives separately. It does not average the scores, combine the checkpoints, or claim that the larger score means the more accurate model for this particular image.")
        st.write("If predictions disagree, photograph multiple leaves and the whole plant in good light, then ask an agriculture extension officer or Krishi Vigyan Kendra to confirm the symptoms. Do not select pesticides from this result alone.")


def render_guide():
    st.markdown("## Disease guide")
    st.write("The two models support different class lists. The guide below combines their labels for reference; similar names do not always mean identical disease definitions.")
    names = sorted(set(DATASETS.keys()))
    for dataset_name in names:
        cfg = DATASETS[dataset_name]
        classes = json.loads((MODEL_DIR / cfg["classes_file"]).read_text(encoding="utf-8"))
        with st.expander(f"{dataset_name} supported classes"):
            for key in classes:
                st.markdown(f"**{pretty_name(key)}** — {class_description(key)}")


def render_about():
    st.markdown("## About KrishiMitra")
    st.write("KrishiMitra is a research-based rice-leaf image classification prototype. It runs two separate MobileNetV3-Small centralized checkpoints trained using two different datasets.")
    st.markdown("### Model details")
    for dataset_name, cfg in DATASETS.items():
        st.markdown(f"**{dataset_name}:** {cfg['description']}")
        st.caption(cfg["metric_note"])
    st.markdown("### Research methods included in prediction")
    st.markdown("""- **Grad-CAM:** visual explanation of image regions influencing a selected class score.
- **MC Dropout:** repeated stochastic forward passes where dropout layers exist; reports score variation.
- **Predictive entropy:** indicates how concentrated or spread the model's class scores are.
- **Mutual information:** exploratory estimate of uncertainty from variation across stochastic passes.
- **OOD screening:** maximum softmax and energy scores are displayed as research indicators, without a validated automatic rejection threshold.
- **Two-model comparison:** outputs are kept separate because the datasets have different class definitions.""")
    st.markdown("### Important limitations")
    st.warning("This prototype is not a certified plant diagnostic service. It does not reliably reject every non-rice image, cannot guarantee a disease diagnosis, and does not provide validated pesticide or treatment recommendations. Model scores are not calibrated probabilities. Grad-CAM is an explanation aid, not proof of lesion detection.")
    st.write("The two checkpoints have different class definitions. Their outputs are compared side by side and are never blended into a single probability.")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--forest:#103d2b;--green:#08763e;--ink:#183b2d;--muted:#63746a;--line:#e1ebe3}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}.stApp{background:#f7faf8}.block-container{max-width:1450px;padding-top:1.1rem;padding-bottom:3rem}
.brand{display:flex;align-items:center;gap:13px;padding:8px 0 18px;border-bottom:1px solid var(--line);margin-bottom:18px}.brand-icon{font-size:29px;background:#e8f5e9;border-radius:50%;padding:8px 12px}.brand-name{font:800 1.65rem Manrope;color:var(--forest)}.brand-sub{color:var(--muted);border-left:1px solid #cad8ce;padding-left:15px}
.hero{background:linear-gradient(110deg,#eef7f0,#e4f1e7);padding:30px;border-radius:20px;margin-bottom:18px}.hero h1{font:800 clamp(2rem,4vw,3rem) Manrope;color:var(--forest);line-height:1.12;margin:.55rem 0 1rem}.hero p{font-size:1.02rem;line-height:1.65;color:#385446}.eyebrow{display:inline-block;background:#d9eddf;color:#155c35;border-radius:20px;padding:7px 13px;font-weight:700;font-size:.83rem}
.feature,.empty{background:#fff;border:1px solid var(--line);border-radius:15px;padding:19px;margin:10px 0}.feature p,.empty p{color:var(--muted);line-height:1.55;margin-bottom:0}.stButton>button{border-radius:10px;font-weight:700;min-height:2.8rem}.stButton>button[kind="primary"]{background:#08763e;border-color:#08763e}[data-testid="stMetric"]{background:#f7faf8;border:1px solid var(--line);border-radius:12px;padding:12px} [data-testid="stFileUploader"]{background:#fff;border:1px dashed #a9cbb4;border-radius:14px;padding:12px}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand'><div class='brand-icon'>🌾</div><div class='brand-name'>KrishiMitra</div><div class='brand-sub'>Dual-model rice leaf disease classification</div></div>", unsafe_allow_html=True)
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"
page = st.radio("Main navigation", ["Home", "Detect Disease", "Disease Guide", "About"], key="nav", horizontal=True, label_visibility="collapsed")

try:
    if page == "Home":
        render_home()
    elif page == "Detect Disease":
        render_detection()
    elif page == "Disease Guide":
        render_guide()
    else:
        render_about()
except Exception as exc:
    st.error("KrishiMitra encountered an unexpected error on this page.")
    with st.expander("Technical details"):
        st.code(str(exc))

st.markdown("<hr><p style='text-align:center;color:#63746a;font-size:.82rem'>KrishiMitra · Dual centralized models · Grad-CAM · Exploratory uncertainty and OOD screening · Confirm crop decisions with an agricultural expert</p>", unsafe_allow_html=True)
