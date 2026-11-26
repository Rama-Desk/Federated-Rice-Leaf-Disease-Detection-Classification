# FedRice-XAI Deployment

This package contains the deployment files for the FedRice-XAI rice-leaf classification demo.

## Recommended prediction model

The app defaults to `models/centralized_best.pth`, the centralized MobileNetV3-Small model trained on Dataset 2. The supplied research notebook reports held-out test accuracy **96.41%** and Macro-F1 **96.22%** for this checkpoint. These are experimental results on the notebook's test split, not a guarantee for arbitrary field images.

The app also includes the five personalized client checkpoints. Select **Personalized federated model** in the sidebar to inspect a client-specific model. These checkpoints are retained for research comparison; they are not assumed to outperform the centralized model.

## Files

- `app.py` — Streamlit interface, prediction, Grad-CAM, and MC Dropout variability
- `model.py` — exact centralized and improved personalized model definitions
- `models/centralized_best.pth` — selected centralized Dataset 2 checkpoint
- `models/client_1.pth` ... `models/client_5.pth` — personalized client checkpoints
- `models/class_names.json` — Dataset 2 class order
- `models/config.json` — model and preprocessing configuration
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python runtime hint

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this folder, preserving the `models/` directory.
3. In Streamlit Community Cloud, create an app from that repository.
4. Set the main file path to `app.py`.

## Important limitations

- The centralized model is the default because it has the strongest reported held-out test metrics among the models listed in the notebook's final comparison.
- Prediction probabilities are not calibrated diagnostic certainty.
- MC Dropout standard deviation is a variability indicator, not a validated medical/agricultural risk score.
- Grad-CAM is an explanation aid, not proof of disease.
- Validated open-set/OOD rejection is not implemented in this deployment.
- Confirm important crop-treatment decisions with a qualified agricultural expert.
