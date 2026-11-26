"""Offline verification for the two bundled centralized checkpoints."""
import json
from pathlib import Path
import torch
from model import build_model

ROOT = Path(__file__).parent
checks = [
    ("Dataset 1", "models/dataset1_centralized_best.pth", "models/dataset1_classes.json", True),
    ("Dataset 2", "models/dataset2_centralized_best.pth", "models/dataset2_classes.json", False),
]
for name, checkpoint_path, class_path, wrapped in checks:
    classes = json.loads((ROOT / class_path).read_text(encoding="utf-8"))
    payload = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=True)
    if wrapped:
        if payload.get("classes") != classes:
            raise RuntimeError(f"{name}: class order mismatch")
        state = payload["model_state_dict"]
    else:
        state = payload
    model = build_model(len(classes))
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    if tuple(output.shape) != (1, len(classes)):
        raise RuntimeError(f"{name}: unexpected output shape {tuple(output.shape)}")
    print(f"PASS: {name}: {len(classes)} classes, output {tuple(output.shape)}")
