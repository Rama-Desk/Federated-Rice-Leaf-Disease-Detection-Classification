# KrishiMitra — Rice Leaf Disease Identification

A Streamlit rice-leaf analysis app with the existing green-and-white reference-style homepage and horizontal navigation: Home, Detect Disease, Disease Guide, About.

## New improvements

- Capture a live image with Streamlit camera input or upload an image.
- Conservative result handling: if the centralized model score, top-two margin, and personalized-client agreement do not meet provisional review checks, the app reports **No reliable single result** instead of presenting a disease as certain.
- Healthy-looking (`normal`) is explained carefully; it does not prove the whole plant is disease-free.
- Top-five classes, Grad-CAM, uncertainty indicators and experimental OOD screening remain available.
- Personalized client models independently corroborate the centralized model; their scores are not averaged.

## Model files

Keep these files in `models/`:

- `config.json`
- `class_names.json`
- `centralized_best.pth`
- `client_1.pth` through `client_5.pth`

Do not rename the files unless you also update the code/configuration.

## Run locally

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
python verify_models.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload the complete folder contents to GitHub, preserving the `models/` and `assets/` directories. Set the main file path to `app.py`. The app needs permission to access the camera in the user's browser for live capture. If camera access is unavailable, use image upload.

## Important accuracy limitation

A model can confidently misclassify a healthy-looking leaf. A confidence score is not a guarantee of correctness. The app now uses provisional abstention checks to reduce unsupported disease claims, but those thresholds have not been calibrated on an independent field-image validation set. The system cannot guarantee correct diagnosis from every image. For a reliable normal-versus-disease decision, evaluate and calibrate the model on a separate, labelled set of healthy and diseased field photographs captured under varied lighting and backgrounds.

OOD scores, entropy, mutual information and MC Dropout are research indicators, not validated clinical/agronomic certainty scores. Grad-CAM is an explanation aid, not lesion segmentation.
