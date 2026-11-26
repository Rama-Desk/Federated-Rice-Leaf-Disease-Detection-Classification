# KrishiMitra — Rice Leaf Disease Classification

A simplified Streamlit application for rice-leaf image classification. The public interface contains Home, Detect Disease, Disease Guide and About only. The research workspace and research dashboards have been removed.

## Features
- Rice-leaf image upload and MobileNetV3-Small inference
- Ten supported classes with plain-language descriptions and field-check guidance
- Top-five class ranking with an explanation of the model scores and the gap between the first two classes
- Grad-CAM heatmap overlay for the selected class
- Clear limitations and advice to confirm important decisions with an agricultural expert

## Run locally
Use Python 3.11 with the pinned packages.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Required files
Keep these files in the supplied locations:
- `models/centralized_best.pth`
- `models/config.json`
- `models/class_names.json`
- All images directly in `assets/`, including `rice_field_hero.jpg` and the disease thumbnails.

Do not change the order in `models/class_names.json` unless it exactly matches the label-to-index mapping used when the checkpoint was trained.

## Important model limitation
The application loads the existing centralized checkpoint; it does not retrain or repair its learned weights. A wrong class prediction cannot be corrected by changing the page text. If the model repeatedly confuses bacterial diseases with Blast, verify the original training label-to-index mapping and evaluate this exact checkpoint on correctly labelled, held-out images from the same dataset. If it fails that test, retrain or replace the checkpoint using the verified mapping and matching preprocessing.

The displayed scores are softmax scores, not calibrated probabilities. The difference between the top two scores is shown as a simple ambiguity indicator, not a validated uncertainty threshold. Grad-CAM highlights regions that influenced the selected class score; it is not lesion segmentation or proof of disease.

This is a research decision-support prototype, not a certified plant diagnostic service. It does not reliably reject every non-rice image and should not be used alone to choose pesticides or treatments. Confirm important crop decisions with an agriculture officer or Krishi Vigyan Kendra.
