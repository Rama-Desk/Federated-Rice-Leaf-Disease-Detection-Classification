# FedRice-XAI — Rice Leaf Analysis Web App

A Streamlit research prototype with a single centralized MobileNetV3-Small prediction model, user-friendly result wording, MC Dropout variability, and Grad-CAM visualization.

## Included
- `app.py` — single-model user interface and inference workflow
- `model.py` — model definition matching the supplied centralized checkpoint
- `models/centralized_best.pth` — selected centralized Dataset 2 checkpoint
- `models/class_names.json` — exact class order
- `models/config.json` — input size, normalization and reported metrics

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
Upload the contents of this folder to the GitHub repository root. Keep the `models` folder. Set the main file to `app.py` and select Python 3.11.

## Important scientific limitations
- The app intentionally exposes one prediction model; client checkpoints are not offered to public users.
- The app does **not** claim validated rice-vs-non-rice or OOD rejection. The research notebook's PlantVillage OOD evaluation was performed on personalized client models, not this centralized deployment checkpoint. A centralized OOD evaluation and validation-based threshold calibration are required before adding a non-rice rejection claim.
- Softmax probability is not calibrated diagnostic certainty. MC Dropout variability is an indicator, not a validated risk score.
- Grad-CAM is an explanation aid, not proof of disease.
- Reported 96.41% accuracy and 96.22% Macro-F1 are notebook results on the held-out Dataset 2 test split, not a guarantee on real-world images.
- This research prototype must not be used alone to select crop treatments.
