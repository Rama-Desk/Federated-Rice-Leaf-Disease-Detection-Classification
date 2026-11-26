# KrishiMitra — Rice Leaf Disease Classification

A Streamlit research prototype for classifying uploaded rice-leaf photographs into the ten supported dataset categories. The interface uses the KrishiMitra name and a clean green-and-white homepage inspired by the supplied reference design.

## Features
- Professional homepage with rice-field hero image and clear disease-detection call to action
- Rice-leaf image upload and MobileNetV3-Small inference
- Ten supported classes and a plain-language disease guide
- Grad-CAM visual explanation
- Prediction scores and exploratory MC Dropout indicators (not a validated rejection rule)
- Research dashboard and methodology pages for centralized and federated experiments
- OOD score evaluation workspace for externally generated validation/test score CSVs

## Run locally
Use Python 3.11 for best compatibility with the pinned PyTorch/Torchvision versions.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model files
Keep `models/centralized_best.pth`, `models/config.json`, and `models/class_names.json` in the `models` folder. Do not rename or move them unless you also update the configuration.

## Research interpretation and limitations
- The public prediction page uses the selected centralized MobileNetV3-Small checkpoint.
- The documented centralized test snapshot reports 95.97% accuracy and 95.69% Macro-F1 on 1,562 held-out images. This is a research result, not a guarantee for new field images.
- FedAvg, FedProx, FedBN and FedPer are shown as research experiments. The recorded federated work is a single-process simulation, not a live multi-farm deployment.
- Grad-CAM highlights image regions that influenced a class score; it is not lesion segmentation or causal proof.
- Quantitative OOD validation and a validated automatic OOD rejection threshold are not included. The OOD page evaluates real score CSVs supplied by the researcher; its example template is illustrative only.
- Temperature-scaling results belong to a separate reported experiment and are not automatically applied to this deployed checkpoint.
- This is a research decision-support prototype, not a certified plant diagnostic service. Confirm important crop-management decisions with an agricultural expert.
