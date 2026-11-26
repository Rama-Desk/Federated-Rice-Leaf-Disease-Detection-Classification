import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CentralizedRiceNet, ImprovedPersonalizedFedRice

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
CONFIG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
CLASS_NAMES = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
NUM_CLASSES = len(CLASS_NAMES)
NUM_CLIENTS = int(CONFIG["num_clients"])
IMAGE_SIZE = int(CONFIG["input_size"][0])
MEAN = CONFIG["normalization"]["mean"]
STD = CONFIG["normalization"]["std"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="FedRice-XAI | Rice Leaf Analysis", page_icon="🌾", layout="wide")

@st.cache_resource(show_spinner="Loading selected model...")
def load_centralized_model():
    model = CentralizedRiceNet(NUM_CLASSES).to(DEVICE)
    state = torch.load(MODEL_DIR / CONFIG["primary_prediction_model"], map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@st.cache_resource(show_spinner="Loading personalized client model...")
def load_personalized_model(client_number: int):
    model = ImprovedPersonalizedFedRice(NUM_CLIENTS, NUM_CLASSES).to(DEVICE)
    state = torch.load(MODEL_DIR / f"client_{client_number}.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def predict_logits(model, tensor, model_kind, client_id):
    if model_kind == "personalized":
        return model(tensor, client_id=client_id)
    return model(tensor)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def gradcam(model, tensor, model_kind, client_id):
    activations, gradients = {}, {}
    target_layer = model.features[-1] if model_kind == "personalized" else model.features[-1]
    def forward_hook(_module, _inputs, output):
        activations["value"] = output
    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = predict_logits(model, tensor, model_kind, client_id)
        class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward()
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_index
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)

def mc_dropout(model, tensor, model_kind, client_id, passes=25):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    probs = []
    with torch.no_grad():
        for _ in range(passes):
            logits = predict_logits(model, tensor, model_kind, client_id)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    model.eval()
    samples = np.asarray(probs)
    return samples.mean(axis=0), samples.std(axis=0)

st.title("FedRice-XAI")
st.caption("Rice leaf disease classification using centralized and personalized federated models")
st.write("The centralized model is selected by default because it achieved the strongest recorded held-out test result in the supplied notebook. Personalized client models are available for research comparison.")

with st.sidebar:
    st.header("Prediction model")
    mode = st.radio("Choose model", ["Best centralized model", "Personalized federated model"], index=0)
    client_number = 1
    if mode == "Personalized federated model":
        client_number = st.selectbox("Client model", list(range(1, NUM_CLIENTS + 1)))
    st.caption(f"Inference device: {DEVICE.type.upper()}")
    st.caption("Input: 224 × 224; ImageNet normalization")

uploaded = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        if mode == "Best centralized model":
            model_kind = "centralized"
            model = load_centralized_model()
            client_id = 0
        else:
            model_kind = "personalized"
            model = load_personalized_model(client_number)
            client_id = client_number - 1

        with st.spinner("Analysing image..."):
            mean_probs, std_probs = mc_dropout(model, tensor, model_kind, client_id)
            cam, predicted_idx = gradcam(model, tensor, model_kind, client_id)

        predicted_name = CLASS_NAMES[predicted_idx]
        confidence = float(mean_probs[predicted_idx])
        variability = float(std_probs[predicted_idx])
        left, right = st.columns(2)
        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Prediction result")
            st.metric("Predicted class", predicted_name.replace("_", " ").title())
            st.metric("Mean model probability", f"{confidence:.1%}")
            st.metric("MC Dropout variability", f"{variability:.4f}")
            if confidence < 0.60:
                st.warning("Low top-class probability. Treat this result as uncertain and seek confirmation from an agricultural expert.")
            elif variability > 0.08:
                st.warning("Predictions vary across dropout passes. Treat this result cautiously.")
            else:
                st.info("This is an AI model prediction, not a confirmed agricultural diagnosis.")

        st.subheader("Class probability ranking")
        for idx in np.argsort(mean_probs)[::-1][:5]:
            st.write(f"**{CLASS_NAMES[idx].replace('_', ' ').title()}** — {mean_probs[idx]:.1%}")
            st.progress(float(mean_probs[idx]))

        st.subheader("Grad-CAM explanation")
        heat = np.uint8(255 * cam)
        heat_rgb = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
        heat_image = Image.fromarray(heat_rgb).resize(image.size)
        st.image(Image.blend(image, heat_image, alpha=0.38), caption="Regions influencing the selected model's prediction", use_container_width=True)
        st.caption("Grad-CAM is an explanation aid, not proof of disease. Probability and dropout variability are not calibrated diagnostic certainty.")
    except Exception as exc:
        st.error(f"Could not analyse this image: {exc}")
else:
    st.info("Choose a model and upload a rice-leaf image to begin.")

with st.expander("Research performance and model information"):
    if mode == "Best centralized model":
        st.write("Recorded Dataset 2 held-out test accuracy: **96.41%**")
        st.write("Recorded Dataset 2 held-out test Macro-F1: **96.22%**")
        st.caption("These are the notebook's reported evaluation results, not a guarantee for every uploaded image or field condition.")
    else:
        st.write("This selection uses one of the five personalized client-specific heads from the Improved Personalized FL experiment.")
    st.write("Architecture: MobileNetV3-Small. The centralized and personalized checkpoints are separate models trained with different procedures.")
    st.write("The current deployment does not make a validated open-set/OOD decision. Upload clear rice-leaf images similar to the training data.")
