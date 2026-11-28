# KrishiMitra — Central Prediction + Federated Cross-Check

Farmer-facing rice leaf disease classification using the supplied Dataset 2 centralized MobileNetV3-Small checkpoint, cross-checked by five selected personalized federated client checkpoints.

## How the single result is chosen

1. The centralized Dataset 2 model produces the primary prediction and top-five class ranking.
2. Five ImprovedPersonalizedFedRice client models independently classify the same image using their corresponding private heads.
3. KrishiMitra displays **one final disease label from the centralized model** and reports how many personalized clients agree.
4. Client confidence values are not averaged with the centralized model because they come from separately trained models and have not been calibrated together.

This is a multi-model cross-check, not a validated accuracy-improving probability ensemble. The supplied research README reports mean validation Macro-F1 0.7996 for the selected round-1 PFL client set; the centralized deployment config reports 96.41% accuracy / 96.22% Macro-F1 on its held-out test snapshot. These are different evaluation summaries and should not be compared as if they were the same test. No claim is made that adding client votes increases accuracy.

## Included models
- `models/centralized_best.pth` — primary classifier.
- `models/client_1.pth` to `client_5.pth` — selected personalized federated client checkpoints.
- Exact ten-class order in `models/class_names.json`.

## Explainability and research indicators
- Grad-CAM for the primary centralized prediction.
- MC Dropout, predictive entropy and mutual information as exploratory uncertainty indicators.
- Maximum Softmax Probability and raw energy score as OOD research indicators.

OOD thresholds are not validated for rice-vs-non-rice rejection. Model scores are not calibrated diagnostic certainty. Grad-CAM is not lesion segmentation or proof of correct reasoning.

## Run
Recommended Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
 pip install -r requirements.txt
streamlit run app.py
```

## Responsible use
KrishiMitra is a research decision-support prototype, not a certified plant diagnostic service. Confirm important crop decisions with an agriculture extension officer or Krishi Vigyan Kendra.
