from ultralytics import YOLO

# Charger le modèle YOLO
model = YOLO("best.pt")

# Export ONNX (opset compatible OpenVINO)
model.export(
    format="onnx",
    opset=11,
    simplify=True,
    dynamic=False
)

print("Export ONNX terminé")
