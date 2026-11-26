from pathlib import Path
import hashlib
import json

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from model import CentralizedRiceNet

st.set_page_config(page_title="KrishiMitra | Rice Leaf Health", page_icon="🌾", layout="wide")

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONFIG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
CLASS_NAMES = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
MODEL_FILE = CONFIG.get("primary_prediction_model", "centralized_best.pth")
TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

DISEASE_INFO = {
    "bacterial_leaf_blight": ("Bacterial leaf blight", "A bacterial disease that can cause water-soaked lesions, yellowing and drying, often progressing from leaf tips or margins.", "Inspect the leaf margins and tips for water-soaked or yellowing areas. Photograph several leaves and the whole plant."),
    "bacterial_leaf_streak": ("Bacterial leaf streak", "Often appears as narrow, elongated, water-soaked streaks that may turn brown.", "Look for fine, parallel streaks between leaf veins. Compare several leaves before drawing a conclusion."),
    "bacterial_panicle_blight": ("Bacterial panicle blight", "Primarily affects panicles and grain development; a leaf-only photograph may not show its main symptoms.", "Inspect the panicle and grain-bearing parts of the plant. A leaf image alone is insufficient to confirm this class."),
    "blast": ("Rice blast", "A fungal disease that can produce spindle-shaped lesions with pale centres and darker margins.", "Check whether lesions are spindle-shaped with grey or pale centres and darker edges. Also inspect nodes and panicles."),
    "brown_spot": ("Brown spot", "Can produce oval or circular brown lesions, sometimes with yellow halos.", "Look for distinct brown spots on multiple leaves. Record whether the spots are round or oval and their distribution."),
    "dead_heart": ("Dead heart", "The central shoot of a young rice tiller dries and may pull out easily; this symptom is not assessed fully from a leaf close-up.", "Inspect the entire tiller and central shoot. Check for dead central leaves and possible stem-borer damage."),
    "downy_mildew": ("Downy mildew", "May cause pale yellow or whitish streaks and abnormal leaf growth.", "Inspect for pale streaks and distorted leaves. Check the plant's growth pattern as well as the photographed leaf."),
    "hispa": ("Rice hispa damage", "Feeding or scraping by rice hispa beetles may leave white streaks or scraped patches on leaves.", "Look for scraped white patches and parallel feeding marks. Inspect the underside of leaves for insect presence."),
    "normal": ("Normal / healthy-looking leaf", "The model places this image in its normal class. This does not prove that the entire plant is disease-free.", "Continue monitoring other leaves, stems and panicles. Retake a clear image if symptoms are visible."),
    "tungro": ("Rice tungro", "A viral disease complex that may cause yellow-orange leaf discoloration, stunting and reduced tillering.", "Inspect overall plant height, leaf colour and tillering. Seek expert confirmation because visual symptoms can overlap."),
}


def nice_name(name):
    return DISEASE_INFO.get(name.lower(), (name.replace('_', ' ').title(), '', ''))[0]


def describe(name):
    return DISEASE_INFO.get(name.lower(), ('', 'No description available.', ''))[1]


def field_checks(name):
    return DISEASE_INFO.get(name.lower(), ('', '', 'Compare the model result with visible symptoms on multiple parts of the plant.'))[2]


@st.cache_resource(show_spinner="Loading the selected centralized model…")
def load_model():
    checkpoint_path = MODEL_DIR / MODEL_FILE
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model = CentralizedRiceNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def gradcam(model, tensor, class_index):
    captured = {}
    target_layer = model.features[-1]

    def hook(_module, _inputs, output):
        captured['activation'] = output
        output.register_hook(lambda grad: captured.__setitem__('gradient', grad))

    handle = target_layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor.detach().clone().requires_grad_(True))
            logits[0, class_index].backward()
        activation = captured.get('activation')
        gradient = captured.get('gradient')
        if activation is None or gradient is None:
            raise RuntimeError('Grad-CAM could not capture model activations and gradients.')
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        if hi - lo < 1e-8:
            raise RuntimeError('The Grad-CAM map is nearly uniform for this image.')
        return np.clip((cam - lo) / (hi - lo), 0, 1)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.eval()


def overlay_cam(image, heatmap):
    base = np.asarray(image.resize((INPUT_SIZE, INPUT_SIZE)).convert('RGB'), dtype=np.float32)
    h = np.clip(heatmap, 0, 1)
    red = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    colour = np.stack([red, green, blue], axis=-1) * 255
    return Image.fromarray(np.uint8(np.clip(0.55 * base + 0.45 * colour, 0, 255)))


def predict(model, image, passes=30):
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    samples = []
    model.eval()  # Keep BatchNorm fixed while enabling only dropout layers.
    try:
        if dropout_layers:
            for layer in dropout_layers:
                layer.train()
            with torch.inference_mode():
                for _ in range(passes):
                    samples.append(torch.softmax(model(tensor), dim=1)[0].cpu().numpy())
        else:
            with torch.inference_mode():
                samples.append(torch.softmax(model(tensor), dim=1)[0].cpu().numpy())
    finally:
        model.eval()

    samples = np.asarray(samples, dtype=np.float64)
    mean_prob = samples.mean(axis=0)
    std_prob = samples.std(axis=0)
    order = np.argsort(mean_prob)[::-1]
    eps = 1e-12
    entropy = float(-(mean_prob * np.log(mean_prob + eps)).sum() / np.log(len(CLASS_NAMES)))
    each_entropy = -(samples * np.log(samples + eps)).sum(axis=1) / np.log(len(CLASS_NAMES))
    expected_entropy = float(each_entropy.mean())
    mutual_info = float(max(0.0, entropy - expected_entropy))
    with torch.no_grad():
        logits = model(tensor)[0].detach().cpu().numpy().astype(np.float64)
    # Numerically stable energy score. This is a raw research indicator, not a calibrated OOD probability.
    max_logit = float(logits.max())
    energy = float(-(max_logit + np.log(np.exp(logits - max_logit).sum())))
    best = int(order[0])
    cam, cam_error = None, None
    try:
        cam = gradcam(model, tensor, best)
    except Exception as exc:
        cam_error = str(exc)
    finally:
        model.eval()
    return {
        'probabilities': mean_prob, 'order': order, 'best': best,
        'entropy': entropy, 'expected_entropy': expected_entropy,
        'mutual_information': mutual_info, 'std': std_prob,
        'max_softmax': float(mean_prob[best]), 'energy': energy,
        'passes': len(samples), 'dropout_count': len(dropout_layers),
        'cam': cam, 'cam_error': cam_error,
        'margin': float(mean_prob[order[0]] - mean_prob[order[1]]),
    }


def render_prediction(image, result):
    best = result['best']
    class_key = CLASS_NAMES[best]
    name = nice_name(class_key)
    score = float(result['probabilities'][best])
    margin = float(result['margin'])
    review_needed = score < 0.50 or margin < 0.10 or result['entropy'] >= 0.65

    st.markdown("""
    <div class="section-eyebrow">ANALYSIS COMPLETE</div>
    <h2 class="result-heading">Rice leaf classification result</h2>
    <p class="section-intro">Review the model's leading class, alternative possibilities and visual explanation below.</p>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment='center')
        with left:
            st.markdown(f"<div class='result-kicker'>TOP PREDICTED CLASS</div><div class='result-title'>{name}</div><div class='result-desc'>{describe(class_key)}</div>", unsafe_allow_html=True)
        with right:
            st.metric('Top model score', f'{score:.1%}', help='Softmax score among the model’s ten supported classes. It is not a calibrated probability of disease.')
        if review_needed:
            st.warning('Needs a closer check. The leading score is modest, the next class is close, or the score distribution is spread out. Compare the alternatives and inspect more leaves.')
        else:
            st.success('One class leads the model’s ten supported categories. Confirm the result against visible symptoms and other parts of the plant.')

    st.markdown('### 1 · Classification breakdown')
    c1, c2, c3 = st.columns(3)
    c1.metric('Lead over second class', f'{margin:.1%}', help='Difference between the first and second ranked class scores. A small gap means the model is less decisive.')
    c2.metric('Predictive entropy', f"{result['entropy']:.3f}", help='Normalized score distribution from 0 to 1. Higher values mean the model spread its scores across more classes.')
    c3.metric('Classes evaluated', str(len(CLASS_NAMES)))

    with st.container(border=True):
        st.markdown('**Top five possible classes**')
        st.caption('Ranked by the mean model score. These are alternatives considered by the classifier, not five confirmed diagnoses.')
        for rank, idx in enumerate(result['order'][:5], start=1):
            idx = int(idx)
            prob = float(result['probabilities'][idx])
            label_col, score_col = st.columns([4, 1])
            label_col.markdown(f"**{rank}. {nice_name(CLASS_NAMES[idx])}**")
            score_col.markdown(f"<div style='text-align:right;font-weight:700;color:#17643b'>{prob:.1%}</div>", unsafe_allow_html=True)
            st.progress(prob)

    st.markdown('### 2 · Visual explanation')
    if result['cam'] is not None:
        with st.container(border=True):
            c1, c2 = st.columns(2, gap='large')
            with c1:
                st.image(image, caption='Uploaded rice leaf image', use_container_width=True)
            with c2:
                st.image(overlay_cam(image, result['cam']), caption=f'Grad-CAM for {name}', use_container_width=True)
            st.info('How to read this: warmer regions contributed more to the selected class score. Grad-CAM highlights model attention; it does not outline disease lesions or prove that the highlighted area is diseased.')
    else:
        st.warning('Grad-CAM could not be generated for this image. The classification remains available.')
        if result['cam_error']:
            st.caption(result['cam_error'])

    st.markdown('### 3 · Uncertainty indicators')
    with st.container(border=True):
        u1, u2, u3 = st.columns(3)
        u1.metric('MC Dropout passes', str(result['passes']))
        u2.metric('Top-score variation', f"{float(result['std'][best]):.4f}")
        u3.metric('Mutual information', f"{result['mutual_information']:.4f}")
        st.caption('MC Dropout repeats prediction with dropout enabled. Predictive entropy measures score spread; top-score variation and mutual information are exploratory indicators. None is a calibrated probability that the result is wrong.')

    st.markdown('### 4 · Out-of-distribution (OOD) screening')
    with st.container(border=True):
        o1, o2 = st.columns(2)
        o1.metric('Maximum Softmax Probability', f"{result['max_softmax']:.3f}")
        o2.metric('Energy score', f"{result['energy']:.3f}")
        if result['max_softmax'] < 0.50 or result['entropy'] >= 0.65:
            st.warning('Review flag raised: the image may be ambiguous or unlike the training examples. This is a provisional screening rule, not a validated OOD rejection.')
        else:
            st.info('No provisional OOD review flag was raised. This does not prove the image is a rice leaf or that the prediction is correct.')
        st.caption('No validated rice-versus-non-rice OOD threshold is available in the supplied research record. MSP and energy are shown as research indicators only.')

    st.markdown('### 5 · Field symptom cross-check')
    with st.container(border=True):
        st.markdown(f"**What this class describes**  \n{describe(class_key)}")
        st.markdown(f"**What to inspect in the field**  \n{field_checks(class_key)}")
        st.caption('For important crop decisions, compare several leaves and consult an agriculture extension officer or Krishi Vigyan Kendra. The app is a decision-support prototype, not a certified diagnosis.')


def render_home():
    left, right = st.columns([1.02, 1.18], gap="small", vertical_alignment="center")
    with left:
        st.markdown("<div class='hero-copy'><span class='eyebrow'>🌾  AI-POWERED RICE CROP SUPPORT</span><h1>Identify and Classify<br>Rice Leaf Diseases</h1><div class='hero-underline'></div><p>Upload a rice leaf image and get a disease classification, with a visual explanation and confidence result. Explore the symptoms of all 10 supported rice leaf categories.</p></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([1.15, .85], gap="small")
        with b1:
            if st.button('Detect Disease Now  →', type='primary', use_container_width=True):
                st.session_state['page'] = 'Detect Disease'
                st.rerun()
        with b2:
            if st.button('ⓘ  Learn More', use_container_width=True):
                st.session_state['page'] = 'About'
                st.rerun()
    with right:
        hero = ASSET_DIR / 'rice_field_hero.jpg'
        if hero.exists():
            st.image(Image.open(hero), use_container_width=True)
    st.markdown("<div class='feature-spacer'></div>", unsafe_allow_html=True)
    features = [
        ('◎', 'Accurate Classification', 'Identify one of 10 rice leaf categories.'),
        ('▱', 'Lightweight & Efficient', 'Fast and resource-friendly MobileNetV3 model.'),
        ('◉', 'Explainable Result', 'See important image areas with Grad-CAM.'),
        ('♢', 'Reliable Prediction', 'Review prediction scores and uncertainty indicators.'),
    ]
    cols = st.columns(4, gap="medium")
    for col, (symbol, title, body) in zip(cols, features):
        with col:
            st.markdown(f"<div class='feature'><div class='feature-icon'>{symbol}</div><div><b>{title}</b><p>{body}</p></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='classes-panel'><div class='classes-heading'><span>🌿 &nbsp;10 Supported Rice Leaf Classes</span><span class='guide-hint'>Explore the Disease Guide →</span></div></div>", unsafe_allow_html=True)
    thumb_files = {
        'bacterial_leaf_blight':'bacterial_leaf_blight.jpg',
        'bacterial_leaf_streak':'bacterial_leaf_streak.jpg',
        'bacterial_panicle_blight':'bacterial_panicle_blight.jpg',
        'blast':'blast.jpg', 'brown_spot':'brown_spot.jpg',
        'dead_heart':'dead_heart.jpg', 'downy_mildew':'downy_mildew.jpg',
        'hispa':'hispa.jpg', 'normal':'normal.jpg', 'tungro':'tungro.jpg'
    }
    cols = st.columns(10, gap="small")
    for i, class_key in enumerate(CLASS_NAMES):
        with cols[i]:
            path = ASSET_DIR / thumb_files.get(class_key, '')
            if path.is_file():
                st.image(Image.open(path), use_container_width=True)
            st.markdown(f"<div class='class-label'>{nice_name(class_key)}</div>", unsafe_allow_html=True)
    if st.button('View Disease Guide  →', use_container_width=False):
        st.session_state['page'] = 'Disease Guide'
        st.rerun()


def render_detection():
    st.markdown("<div class='section-eyebrow'>IMAGE-BASED RICE HEALTH CHECK</div><h2 class='result-heading'>Detect rice leaf disease</h2><p class='section-intro'>Upload one clear, unfiltered photograph. The centralized MobileNetV3-Small model will classify it and generate an explanation.</p>", unsafe_allow_html=True)
    upload = st.file_uploader('Choose a rice-leaf image', type=['jpg', 'jpeg', 'png', 'webp'], help='Use a focused photograph in natural light. Avoid screenshots, collages and heavily edited images.')
    if not upload:
        st.markdown("<div class='empty'><b>Ready for image analysis</b><p>Upload a clear rice-leaf photograph to see the classification and explanations.</p></div>", unsafe_allow_html=True)
        return
    token = hashlib.sha256(upload.getvalue()).hexdigest()
    try:
        image = ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
    except Exception:
        st.error('This image could not be opened. Please upload a standard JPG or PNG photograph.')
        return
    if min(image.size) < 120:
        st.error('The image is too small. Please upload a clearer photograph with at least 120 pixels on each side.')
        return
    st.image(image, caption='Uploaded image', width=460)
    if st.button('Analyse image', type='primary', use_container_width=True):
        try:
            with st.spinner('Analysing image and generating explanations…'):
                model = load_model()
                result = predict(model, image)
            st.session_state['prediction'] = {'token': token, 'result': result}
        except Exception as exc:
            st.error('The model could not complete the prediction. Check the checkpoint and package installation.')
            with st.expander('Technical error details'):
                st.code(str(exc))
            return
    stored = st.session_state.get('prediction')
    if stored and stored.get('token') == token:
        st.divider()
        render_prediction(image, stored['result'])


def render_guide():
    st.markdown("<div class='section-eyebrow'>REFERENCE LIBRARY</div><h2 class='result-heading'>Rice disease guide</h2>", unsafe_allow_html=True)
    st.write('These are the ten classes supported by the primary centralized model.')
    for class_key in CLASS_NAMES:
        with st.expander(nice_name(class_key)):
            st.write(describe(class_key))
            st.markdown('**What to inspect:** ' + field_checks(class_key))


def render_about():
    st.markdown("<div class='section-eyebrow'>PROJECT INFORMATION</div><h2 class='result-heading'>About KrishiMitra</h2>", unsafe_allow_html=True)
    st.write('KrishiMitra is a research-based rice-leaf image classification prototype. Its public prediction flow uses the centralized MobileNetV3-Small checkpoint trained on Dataset 2.')
    st.markdown('### Selected model')
    st.write('**Architecture:** MobileNetV3-Small')
    st.write('**Dataset:** Dataset 2 — paddy-disease-classification')
    st.write('**Input:** 224 × 224 RGB image with ImageNet normalization')
    st.write('**Supported classes:** ' + ', '.join(nice_name(x) for x in CLASS_NAMES))
    st.markdown('### Evaluation record and metric consistency')
    st.write('The supplied deployment configuration reports 96.41% test accuracy and 96.22% Macro-F1. A separate supplied project snapshot reports 95.97% test accuracy and 95.69% Macro-F1 for this same checkpoint. Because these records conflict, the app does not present either value as a verified final result.')
    st.warning('Re-run evaluation on the exact held-out test split and preserve the resulting classification report before publishing one definitive accuracy figure. Neither result guarantees performance on field photographs or images from another source.')
    st.markdown('### Research methods in the prediction page')
    st.markdown('- **Grad-CAM:** visual explanation of image regions influencing the selected class score.\n- **MC Dropout:** repeated stochastic inference using the model dropout layer.\n- **Predictive entropy and mutual information:** exploratory uncertainty indicators.\n- **OOD screening:** maximum softmax probability and energy score as research indicators.')
    st.warning('OOD thresholds have not been validated against an external rice/non-rice test set in the supplied research record. The app therefore shows provisional screening indicators, not a proven OOD detector. Model scores are not calibrated probabilities; Grad-CAM is not proof of lesion detection. Confirm important crop decisions with an agriculture extension officer or Krishi Vigyan Kendra.')


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root{--forest:#073b2b;--green:#08763e;--ink:#123c2e;--muted:#52645b;--line:#e1ebe4;--pale:#eaf5ec}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:linear-gradient(180deg,#fbfdfb 0%,#f3f8f4 100%)}
.block-container{max-width:1600px;padding:0 2.2rem 2.5rem}
.brand{display:flex;align-items:center;gap:14px;padding:9px 0 14px;border-bottom:1px solid #e5ece7;margin:0 0 0;min-height:68px}
.brand-icon{font-size:34px;background:transparent;padding:0}.brand-name{font:800 1.65rem Manrope;color:var(--forest);letter-spacing:-.04em}.brand-sub{color:#173c30;border-left:1px solid #c9d8ce;padding-left:20px;font-size:1rem}
/* top navigation */
div[role="radiogroup"]{display:flex;justify-content:flex-end;gap:10px;margin-top:-57px;margin-bottom:20px;position:relative;z-index:5;min-height:45px}
div[role="radiogroup"] label{border:1px solid transparent;border-radius:13px;padding:8px 17px;background:transparent;color:#123c2e;font-weight:700;white-space:nowrap}
div[role="radiogroup"] label:has(input:checked){background:#21864e;color:white;box-shadow:0 4px 12px #1d744022}
div[role="radiogroup"] label p{font-size:.98rem}
div[role="radiogroup"] input{display:none}
.hero-copy{padding:28px 28px 20px 24px;min-height:350px;display:flex;flex-direction:column;justify-content:center;background:radial-gradient(ellipse at 85% 12%,#e5f4e7 0%,#f8fcf8 55%,#fff 100%);border-radius:20px 0 0 20px}
.hero-copy h1{font:800 clamp(2.2rem,3.5vw,3.45rem) Manrope;color:#073b2b;line-height:1.03;letter-spacing:-.045em;margin:17px 0 7px}
.hero-copy p{font-size:1.12rem;line-height:1.48;color:#153c2f;max-width:620px;margin:18px 0 14px}
.hero-underline{height:5px;width:285px;border-radius:100%;background:#79c65d;transform:rotate(-1deg);margin-top:0}
.eyebrow{display:inline-block;align-self:flex-start;background:#e0f2e2;color:#064a2c;border-radius:24px;padding:9px 16px;font-weight:800;font-size:.94rem}
div[data-testid="stImage"] img{border-radius:16px;object-fit:cover}
.feature-spacer{height:16px}
.feature{height:100%;min-height:112px;background:#fff;border:1px solid #edf2ee;border-radius:20px;padding:18px 17px;display:flex;align-items:center;gap:15px;box-shadow:0 5px 20px #153b2b08;margin:0}
.feature-icon{flex:0 0 66px;height:66px;border-radius:18px;background:#edf7ef;color:#08763e;display:flex;align-items:center;justify-content:center;font-size:2.2rem;font-weight:700}
.feature b{font-size:1.02rem;color:#0b3829}.feature p{font-size:.91rem;color:#52645b;line-height:1.4;margin:5px 0 0}
.classes-panel{background:#fff;border:1px solid #dceae0;border-radius:24px 24px 0 0;padding:19px 24px 0;margin-top:24px}
.classes-heading{display:flex;justify-content:space-between;align-items:center;font:800 1.55rem Manrope;color:#0b3c2c;margin-bottom:13px}.guide-hint{font:700 .9rem 'DM Sans';color:#17643b;background:#edf7ef;border-radius:20px;padding:9px 16px}
.class-label{background:#fff;border:1px solid #e0eae3;border-radius:0 0 13px 13px;text-align:center;min-height:49px;padding:7px 3px;font-weight:700;font-size:.78rem;line-height:1.2;color:#153b2e;box-shadow:0 3px 8px #0d3b2008}
.result-card{background:linear-gradient(110deg,#edf7ef,#e5f2e8);border:1px solid #d8e9dc;border-radius:16px;padding:22px;margin:10px 0 18px}.result-kicker{font-size:.75rem;letter-spacing:.08em;font-weight:700;color:#39704f}.result-title{font:800 1.8rem Manrope;color:#103d2b;margin:5px 0}.result-desc{color:#385446;line-height:1.55}
.stButton>button{border-radius:13px;font-weight:800;min-height:2.85rem;border:1px solid #e2ece5}.stButton>button[kind="primary"]{background:#08763e;border-color:#08763e;color:#fff;box-shadow:0 5px 12px #08763e22}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}[data-testid="stFileUploader"]{background:#fff;border:1px dashed #a9cbb4;border-radius:14px;padding:12px}
h2,h3{font-family:'Manrope',sans-serif;color:#103d2b}footer{visibility:hidden}#MainMenu{visibility:hidden}

.section-eyebrow{display:inline-block;background:#e8f5eb;color:#17643b;border-radius:20px;padding:6px 12px;font-size:.75rem;font-weight:800;letter-spacing:.07em;margin:8px 0}
.result-heading{font:800 clamp(1.55rem,2.3vw,2.2rem) Manrope;color:#103d2b;margin:5px 0 2px}
.section-intro{color:#607168;margin:0 0 14px}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:#e0ebe3!important;border-radius:18px!important;background:rgba(255,255,255,.78)}
[data-testid="stMetricValue"]{color:#0b6338;font-weight:800}
[data-testid="stProgressBar"]>div{background:#e7f0e9}
[data-testid="stProgressBar"]>div>div{background:#23864d}

@media(max-width:900px){.block-container{padding:0 1rem 2rem}div[role="radiogroup"]{margin:0;justify-content:flex-start;flex-wrap:wrap}.brand-sub{display:none}.hero-copy{min-height:0;padding:22px}.classes-heading{font-size:1.15rem}.class-label{font-size:.65rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand'><div class='brand-icon'>🌾</div><div class='brand-name'>KrishiMitra</div><div class='brand-sub'>Rice Leaf Disease Classification</div></div>", unsafe_allow_html=True)
NAV_LABELS = {'Home': '⌂  Home', 'Detect Disease': '◎  Detect Disease', 'Disease Guide': '▤  Disease Guide', 'About': 'ⓘ  About'}
if 'page' not in st.session_state or st.session_state['page'] not in NAV_LABELS:
    st.session_state['page'] = 'Home'
page = st.radio('Main navigation', list(NAV_LABELS), format_func=lambda item: NAV_LABELS[item], key='page', horizontal=True, label_visibility='collapsed')

try:
    if page == 'Home':
        render_home()
    elif page == 'Detect Disease':
        render_detection()
    elif page == 'Disease Guide':
        render_guide()
    else:
        render_about()
except Exception as exc:
    st.error('KrishiMitra encountered an unexpected error on this page.')
    with st.expander('Technical error details'):
        st.code(str(exc))

st.markdown("<hr><p style='text-align:center;color:#63746a;font-size:.82rem'>KrishiMitra · Centralized MobileNetV3-Small · Grad-CAM · MC Dropout · Exploratory OOD screening</p>", unsafe_allow_html=True)
