# KrishiMitra — Dual-Model Rice Leaf Analysis

A Streamlit application using the same clean green-and-white KrishiMitra website design: branded header, split hero section, feature cards, image upload, and simple Home / Detect Disease / Disease Guide / About navigation.

## Included centralized models

- **Dataset 1:** `models/dataset1_centralized_best.pth`
- **Dataset 2:** `models/dataset2_centralized_best.pth`

Each checkpoint retains its own class list and class order. Their scores are not averaged or treated as directly comparable probabilities.

## Research methods integrated into prediction

- **Grad-CAM:** generates a heatmap for each model's selected class.
- **MC Dropout:** uses repeated stochastic passes when dropout layers are present. The supplied MobileNetV3-Small architecture includes a dropout layer.
- **Predictive entropy:** normalized measure of how spread the class-score distribution is.
- **Top-class variation:** standard deviation of the selected class score across MC-dropout passes.
- **Mutual information:** exploratory estimate of uncertainty across stochastic passes.
- **OOD research indicators:** maximum softmax probability and energy score are shown for each model. A transparent review warning is raised for low maximum score or high predictive entropy.

### OOD and uncertainty limitations

The project record supplied with this application states that quantitative PlantVillage OOD evaluation is pending and no validated OOD threshold is available. Therefore:

- OOD scores are **research screening indicators**, not a validated rice/non-rice detector.
- The review warning uses provisional heuristic cut-offs (maximum softmax below 0.50 or normalized entropy at/above 0.65). These values are not empirically calibrated and must not be presented as research performance.
- Energy score is shown as a raw diagnostic; no threshold is applied.
- MC Dropout variation, predictive entropy and mutual information are exploratory and are not calibrated probabilities of error.
- A “no warning” result does not prove that an image is in-distribution or correctly classified.

## Prediction interpretation

Each model shows its own predicted class, plain-language disease description, top-five alternatives, score margin, reliability indicators and Grad-CAM. If models disagree, the app explicitly recommends verification rather than choosing the larger score. Different dataset class definitions are preserved.

## Run locally

Recommended: Python 3.10 or 3.11.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload the complete project contents to GitHub. Keep `app.py`, `model.py`, `requirements.txt`, `models/` and `assets/` at the repository root. Set the app entry point to `app.py`.

## Verification

Run:

```bash
python verify_models.py
```

The script checks both checkpoint files, their class mappings and expected ten-class model outputs.

## Responsible use

This is a research decision-support prototype, not a certified plant diagnostic service. It does not provide pesticide recommendations. Confirm important crop decisions with an agriculture extension officer or Krishi Vigyan Kendra.
