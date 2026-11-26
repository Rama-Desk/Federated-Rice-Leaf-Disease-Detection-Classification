# KrishiMitra — Best Centralized Model Edition

A farmer-facing Streamlit application using the same clean green-and-white KrishiMitra website design: leaf brand row, split hero section, feature cards, ten supported class thumbnails, and simple Home / Detect Disease / Disease Guide / About navigation.

## Primary model selected

- **Architecture:** MobileNetV3-Small
- **Dataset:** Dataset 2 — `paddy-disease-classification`
- **Checkpoint:** `models/centralized_best.pth`
- **Classes:** 10, stored in `models/class_names.json` in the exact training order
- **Input:** 224 × 224 RGB, ImageNet mean and standard deviation

This is the recommended public prediction model because the supplied FedRice-XAI deployment configuration identifies this centralized checkpoint as the primary model and reports it as the strongest held-out candidate in that deployment snapshot. The Dataset 1 model is not blended into this prediction path because its label definitions differ.

## Evaluation metric discrepancy — read before publishing results

The supplied project files contain conflicting evaluation values for this same Dataset 2 checkpoint:

- FedRice-XAI deployment `config.json`: **96.41% accuracy / 96.22% Macro-F1**.
- A separate supplied KrishiMitra project snapshot: **95.97% accuracy / 95.69% Macro-F1**.

The checkpoint SHA-256 identity matches between the deployment archive and the dual-model archive, but the reported metrics differ. This package therefore discloses the discrepancy rather than claiming one definitive score. Re-run evaluation against the exact held-out test split and save the classification report before presenting a final research metric. Neither score guarantees accuracy on real field photographs.

## Research methods included in prediction

- **Grad-CAM:** generates a class-specific heatmap from the final MobileNetV3 feature block. It explains influential regions; it is not lesion segmentation or proof of correct reasoning.
- **MC Dropout:** performs 30 stochastic forward passes with the classifier dropout layer enabled while BatchNorm stays in evaluation mode.
- **Predictive entropy:** normalized measure of how spread the mean class-score distribution is.
- **Mutual information:** exploratory estimate of uncertainty across stochastic passes.
- **Top-class variation:** standard deviation of the selected class score across MC passes.
- **OOD indicators:** Maximum Softmax Probability and raw energy score are displayed.

### OOD and uncertainty limitations

The supplied research record does not provide a validated rice-versus-non-rice OOD threshold or completed external OOD calibration. The app uses a clearly labelled provisional review flag (maximum softmax below 0.50 or normalized entropy at/above 0.65). These cut-offs are heuristic and are not empirically calibrated. They must not be presented as a validated OOD detector or deployment rejection guarantee. MC Dropout and entropy outputs are exploratory and are not calibrated probabilities of error.

## Run locally

Recommended: Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python verify_model.py
streamlit run app.py
```

## GitHub / Streamlit deployment

Upload the complete folder contents to GitHub, preserving `app.py`, `model.py`, `requirements.txt`, `models/` and `assets/`. Set the Streamlit entry point to `app.py`.

## Responsible use

KrishiMitra is a research decision-support prototype, not a certified plant diagnostic service. It does not provide pesticide recommendations. Confirm important crop decisions with an agriculture extension officer or Krishi Vigyan Kendra.
