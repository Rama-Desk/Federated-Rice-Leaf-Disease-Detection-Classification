"""Verify the centralized and five personalized Dataset 2 checkpoints."""
from pathlib import Path
import json
import torch
from model import CentralizedRiceNet, ImprovedPersonalizedFedRice

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
classes = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))

primary_path = MODEL_DIR / config.get("primary_prediction_model", "centralized_best.pth")
primary = CentralizedRiceNet(num_classes=len(classes))
primary.load_state_dict(torch.load(primary_path, map_location="cpu", weights_only=True), strict=True)
primary.eval()
with torch.inference_mode():
    out = primary(torch.zeros(1, 3, 224, 224))
assert tuple(out.shape) == (1, len(classes)) and torch.isfinite(out).all()
print(f"PASS: centralized checkpoint; {len(classes)} classes")

for client_id in range(1, 6):
    path = MODEL_DIR / f"client_{client_id}.pth"
    model = ImprovedPersonalizedFedRice(num_clients=5, num_classes=len(classes))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 224, 224), client_id=client_id - 1)
    assert tuple(out.shape) == (1, len(classes)) and torch.isfinite(out).all()
    print(f"PASS: personalized client {client_id}; {len(classes)} classes")
print("Class order:", ", ".join(classes))
