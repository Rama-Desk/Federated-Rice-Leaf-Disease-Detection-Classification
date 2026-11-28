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

from model import CentralizedRiceNet, ImprovedPersonalizedFedRice

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


@st.cache_resource(show_spinner="Loading five personalized research models…")
def load_personalized_models():
    """Load the five selected round-1 client models for consensus checking."""
    models = []
    for client_id in range(1, 6):
        path = MODEL_DIR / f"client_{client_id}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Personalized client checkpoint missing: {path}")
        model = ImprovedPersonalizedFedRice(num_clients=5, num_classes=len(CLASS_NAMES)).to(DEVICE)
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models


def run_client_consensus(image):
    """Return one vote per personalized client; do not average incomparable confidence scores."""
    models = load_personalized_models()
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    votes = []
    with torch.inference_mode():
        for client_number, model in enumerate(models, start=1):
            # Each client checkpoint is evaluated through its corresponding private head.
            logits = model(tensor, client_id=client_number - 1)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            votes.append({
                "client": client_number,
                "class_index": int(np.argmax(probs)),
                "top_score": float(np.max(probs)),
            })
    return votes


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
    client_votes = run_client_consensus(image)
    client_agree = sum(v['class_index'] == best for v in client_votes)
    return {
        'probabilities': mean_prob, 'order': order, 'best': best,
        'entropy': entropy, 'expected_entropy': expected_entropy,
        'mutual_information': mutual_info, 'std': std_prob,
        'max_softmax': float(mean_prob[best]), 'energy': energy,
        'passes': len(samples), 'dropout_count': len(dropout_layers),
        'cam': cam, 'cam_error': cam_error,
        'margin': float(mean_prob[order[0]] - mean_prob[order[1]]),
        'client_votes': client_votes, 'client_agreement': client_agree,
    }


def render_prediction(image, result):
    best = result['best']
    class_key = CLASS_NAMES[best]
    name = nice_name(class_key)
    score = float(result['probabilities'][best])

    st.markdown('### Prediction result')
    st.markdown(f"<div class='result-card'><div class='result-kicker'>MODEL'S TOP CLASS</div><div class='result-title'>{name}</div><div class='result-desc'>{describe(class_key)}</div></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric('Top model score', f'{score:.1%}', help='Softmax score within the ten classes. It is not a calibrated probability that the plant has this disease.')
    m2.metric('Lead over second class', f"{result['margin']:.1%}", help='Difference between the top two mean class scores. A small gap means the model finds the leading classes difficult to separate.')
    m3.metric('Predictive entropy', f"{result['entropy']:.3f}", help='Normalized score spread from 0 to 1. Higher values indicate a more spread-out prediction distribution.')

    if score < 0.50 or result['margin'] < 0.10:
        st.warning('Tentative classification: the leading score is low or close to the second class. Check the alternative classes and photograph the plant again in natural light.')
    else:
        st.info('The model has a clear leading class within its supported categories. This is not proof of a correct field diagnosis.')

    st.markdown('#### Cross-check with personalized federated models')
    agree = int(result.get('client_agreement', 0))
    total = len(result.get('client_votes', []))
    st.metric('Personalized clients agreeing with the main prediction', f'{agree} / {total}')
    if agree >= 4:
        st.success('Strong model agreement: the centralized model and most personalized client models selected the same disease class.')
    elif agree >= 2:
        st.info('Partial model agreement: some personalized clients support the main prediction, while others differ. Review the image and symptoms.')
    else:
        st.warning('Model disagreement: personalized clients do not strongly support the main prediction. Treat the result as tentative and seek field verification.')
    with st.expander('View individual model votes'):
        for vote in result.get('client_votes', []):
            st.write(f"Client {vote['client']}: **{nice_name(CLASS_NAMES[vote['class_index']])}** · top score {vote['top_score']:.1%}")
        st.caption('The final disease label remains the primary centralized model prediction. Client models are used as independent corroboration, not averaged into a synthetic confidence score. This consensus check has not been validated as an accuracy-improving ensemble.')

    st.markdown('#### What this result means')
    st.write(describe(class_key))
    st.markdown('#### What to inspect in the field')
    st.write(field_checks(class_key))

    st.markdown('#### Top five classes considered')
    for rank, idx in enumerate(result['order'][:5], start=1):
        idx = int(idx)
        prob = float(result['probabilities'][idx])
        st.write(f"**{rank}. {nice_name(CLASS_NAMES[idx])}** — {prob:.1%}")
        st.progress(prob)

    st.markdown('#### Visual explanation — Grad-CAM')
    if result['cam'] is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption='Original uploaded image', use_container_width=True)
        with c2:
            st.image(overlay_cam(image, result['cam']), caption=f'Areas influencing the {name} score', use_container_width=True)
        st.caption('Warm colours indicate image regions that contributed more to the selected class score. Grad-CAM is an explanation aid; it does not segment lesions or prove the model focused on the correct symptom.')
    else:
        st.warning('Grad-CAM could not be generated for this image. The classification result remains available.')
        if result['cam_error']:
            st.caption(result['cam_error'])

    st.markdown('#### Uncertainty analysis')
    u1, u2, u3 = st.columns(3)
    u1.metric('MC Dropout passes', str(result['passes']))
    u2.metric('Top-class score variation', f"{float(result['std'][best]):.4f}")
    u3.metric('Mutual information', f"{result['mutual_information']:.4f}")
    st.caption('MC Dropout repeats inference with dropout enabled. Predictive entropy describes score spread; mutual information and score variation are exploratory uncertainty indicators, not calibrated error probabilities.')

    st.markdown('#### Out-of-distribution (OOD) screening')
    o1, o2 = st.columns(2)
    o1.metric('Maximum Softmax Probability', f"{result['max_softmax']:.3f}")
    o2.metric('Energy score', f"{result['energy']:.3f}")
    if result['max_softmax'] < 0.50 or result['entropy'] >= 0.65:
        st.warning('OOD / ambiguity review flag: this image has a low top score or a diffuse score distribution. It may be ambiguous or unlike the training examples. This is a provisional heuristic, not a validated OOD rejection decision.')
    else:
        st.info('No provisional OOD review flag was raised. This does not prove that the image belongs to the training distribution or that the prediction is correct.')
    st.caption('The supplied research package does not provide a validated rice-versus-non-rice OOD threshold or completed external OOD calibration. MSP and energy are shown as research indicators only; KrishiMitra does not claim reliable automatic rejection of every non-rice image.')


def render_home():
    left, right = st.columns([1.05, 0.95], gap='large', vertical_alignment='center')
    with left:
        st.markdown("<div class='hero'><span class='eyebrow'>AI-assisted rice crop support</span><h1>Identify and understand rice leaf diseases</h1><p>Upload a clear rice-leaf photograph. KrishiMitra analyses it with the selected centralized MobileNetV3-Small model and explains the prediction with alternative classes, Grad-CAM and research reliability indicators.</p></div>", unsafe_allow_html=True)
        if st.button('Detect Disease Now', type='primary', use_container_width=True):
            st.session_state['page'] = 'Detect Disease'
            st.rerun()
        st.caption('Research prototype · Decision support only')
    with right:
        hero = ASSET_DIR / 'rice_field_hero.jpg'
        if hero.exists():
            st.image(Image.open(hero), use_container_width=True)
    st.markdown('### What KrishiMitra provides')
    features = [
        ('Accurate classification', 'One selected centralized model with a documented held-out evaluation result.'),
        ('Lightweight and efficient', 'MobileNetV3-Small designed for compact image classification.'),
        ('Explainable result', 'Grad-CAM heatmap, top-five alternatives and plain-language descriptions.'),
        ('Reliability indicators', 'MC Dropout, entropy, mutual information and exploratory OOD screening.'),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, features):
        with col:
            st.markdown(f"<div class='feature'><b>{title}</b><p>{body}</p></div>", unsafe_allow_html=True)
    st.markdown('### 10 supported classes')
    thumb_files = {
        'bacterial_leaf_blight':'bacterial_leaf_blight.jpg',
        'bacterial_leaf_streak':'bacterial_leaf_streak.jpg',
        'bacterial_panicle_blight':'bacterial_panicle_blight.jpg',
        'blast':'blast.jpg', 'brown_spot':'brown_spot.jpg',
        'dead_heart':'dead_heart.jpg', 'downy_mildew':'downy_mildew.jpg',
        'hispa':'hispa.jpg', 'normal':'normal.jpg', 'tungro':'tungro.jpg'
    }
    cols = st.columns(5)
    for i, class_key in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            path = ASSET_DIR / thumb_files.get(class_key, '')
            if path.is_file():
                st.image(Image.open(path), use_container_width=True)
            st.caption(nice_name(class_key))


def render_detection():
    st.markdown('## Detect rice leaf disease')
    st.write('Upload one clear, unfiltered image. The Dataset 2 centralized model makes the main prediction; five personalized federated client models independently cross-check it.')
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
    st.markdown('## Rice disease guide')
    st.write('These are the ten classes supported by the primary centralized model.')
    for class_key in CLASS_NAMES:
        with st.expander(nice_name(class_key)):
            st.write(describe(class_key))
            st.markdown('**What to inspect:** ' + field_checks(class_key))


def render_about():
    st.markdown('## About KrishiMitra')
    st.write('KrishiMitra is a research-based rice-leaf image classification prototype. Its public prediction flow uses the centralized MobileNetV3-Small checkpoint trained on Dataset 2.')
    st.markdown('### Selected model')
    st.write('**Architecture:** MobileNetV3-Small')
    st.write('**Dataset:**paddy-disease-classification')
    st.write('**Input:** 224 × 224 RGB image with ImageNet normalization')
    st.write('**Supported classes:** ' + ', '.join(nice_name(x) for x in CLASS_NAMES))
    st.write('**Research cross-check:** Five ImprovedPersonalizedFedRice client checkpoints, each evaluated with its corresponding private head.')
    st.markdown('### Evaluation record and metric consistency')
    st.write('The supplied deployment configuration reports 96.41% test accuracy and 96.22% Macro-F1. A separate supplied project snapshot reports 95.97% test accuracy and 95.69% Macro-F1 for this same checkpoint. Because these records conflict, the app does not present either value as a verified final result.')
    st.warning('Re-run evaluation on the exact held-out test split and preserve the resulting classification report before publishing one definitive accuracy figure. Neither result guarantees performance on field photographs or images from another source.')
    st.markdown('### Research methods in the prediction page')
    st.markdown('- **Grad-CAM:** visual explanation of image regions influencing the selected class score.\n- **MC Dropout:** repeated stochastic inference using the model dropout layer.\n- **Predictive entropy and mutual information:** exploratory uncertainty indicators.\n- **OOD screening:** maximum softmax probability and energy score as research indicators.')
    st.warning('OOD thresholds have not been validated against an external rice/non-rice test set in the supplied research record. The app therefore shows provisional screening indicators, not a proven OOD detector. Model scores are not calibrated probabilities; Grad-CAM is not proof of lesion detection. Confirm important crop decisions with an agriculture extension officer or Krishi Vigyan Kendra.')


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--forest:#103d2b;--green:#08763e;--ink:#183b2d;--muted:#63746a;--line:#e1ebe3}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}.stApp{background:#f7faf8}.block-container{max-width:1450px;padding-top:1.1rem;padding-bottom:3rem}
.brand{display:flex;align-items:center;gap:13px;padding:8px 0 18px;border-bottom:1px solid var(--line);margin-bottom:18px}.brand-icon{font-size:29px;background:#e8f5e9;border-radius:50%;padding:8px 12px}.brand-name{font:800 1.65rem Manrope;color:var(--forest)}.brand-sub{color:var(--muted);border-left:1px solid #cad8ce;padding-left:15px}
.hero{background:linear-gradient(110deg,#eef7f0,#e4f1e7);padding:30px;border-radius:20px;margin-bottom:18px}.hero h1{font:800 clamp(2rem,4vw,3rem) Manrope;color:var(--forest);line-height:1.12;margin:.55rem 0 1rem}.hero p{font-size:1.02rem;line-height:1.65;color:#385446}.eyebrow{display:inline-block;background:#d9eddf;color:#155c35;border-radius:20px;padding:7px 13px;font-weight:700;font-size:.83rem}
.feature,.empty{background:#fff;border:1px solid var(--line);border-radius:15px;padding:19px;margin:10px 0}.feature p,.empty p{color:var(--muted);line-height:1.55;margin-bottom:0}.result-card{background:linear-gradient(110deg,#edf7ef,#e5f2e8);border:1px solid #d8e9dc;border-radius:16px;padding:22px;margin:10px 0 18px}.result-kicker{font-size:.75rem;letter-spacing:.08em;font-weight:700;color:#39704f}.result-title{font:800 1.8rem Manrope;color:#103d2b;margin:5px 0}.result-desc{color:#385446;line-height:1.55}
.stButton>button{border-radius:10px;font-weight:700;min-height:2.8rem}.stButton>button[kind="primary"]{background:#08763e;border-color:#08763e}[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}[data-testid="stFileUploader"]{background:#fff;border:1px dashed #a9cbb4;border-radius:14px;padding:12px}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand'><div class='brand-icon'>🌾</div><div class='brand-name'>KrishiMitra</div><div class='brand-sub'>Rice leaf disease identification and explanation</div></div>", unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
page = st.radio('Main navigation', ['Home', 'Detect Disease', 'Disease Guide', 'About'], key='page', horizontal=True, label_visibility='collapsed')

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

st.markdown("<hr><p style='text-align:center;color:#63746a;font-size:.82rem'>KrishiMitra · Centralized + Personalized Model Cross-check · Grad-CAM · MC Dropout · Exploratory OOD screening</p>", unsafe_allow_html=True)
