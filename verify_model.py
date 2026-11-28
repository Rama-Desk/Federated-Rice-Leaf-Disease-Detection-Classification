"""Verify the primary Dataset 2 checkpoint and model output shape."""
from pathlib import Path
import json
import torch
from model import CentralizedRiceNet

ROOT = Path(__file__).parent
classes = json.loads((ROOT / 'models' / 'class_names.json').read_text(encoding='utf-8'))
config = json.loads((ROOT / 'models' / 'config.json').read_text(encoding='utf-8'))
checkpoint = ROOT / 'models' / config['primary_prediction_model']
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
model = CentralizedRiceNet(len(classes))
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 3, 224, 224))
assert tuple(output.shape) == (1, len(classes)), output.shape
assert torch.isfinite(output).all()
print(f'PASS: {config["architecture"]}; {len(classes)} classes; output={tuple(output.shape)}')
print('Class order:', ', '.join(classes))
