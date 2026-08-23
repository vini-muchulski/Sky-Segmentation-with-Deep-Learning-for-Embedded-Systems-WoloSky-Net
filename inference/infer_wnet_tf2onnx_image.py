import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ----- caminhos -----


image_input_path = r"inference/imgs/drone (1).jpg"

#IMG_SZ = (320, 320)  
#IMG_SZ = (256, 256)
IMG_SZ = (224, 224)
#IMG_SZ = (160, 160)


model_path = r"weights/v2_W-Net_U2Net_Sam3_320x320_infer.onnx"


onnx_model_path  = Path(model_path)

image_input_path = Path(image_input_path)

prob_threshold = 0.5

assert onnx_model_path.exists(),  f"Modelo não encontrado: {onnx_model_path}"
assert image_input_path.exists(), f"Imagem não encontrada: {image_input_path}"

# ----- sessão ORT -----
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(str(onnx_model_path), sess_opts, providers=["CPUExecutionProvider"])

# ins/outs
inputs  = session.get_inputs()
outputs = session.get_outputs()

input_name  = inputs[0].name
input_shape = inputs[0].shape
print("input", input_shape)

# tente escolher a cabeça fundida se houver várias saídas
if len(outputs) > 1:
    fused_idx = None
    for i, o in enumerate(outputs):
        n = (o.name or "").lower()
        if "output_fused" in n or "fused" in n:
            fused_idx = i
            break
    if fused_idx is None:
        fused_idx = 0
    output_name = outputs[fused_idx].name
else:
    output_name = outputs[0].name

print("output(s):", [(o.name, o.shape) for o in outputs])
print("usando saída:", output_name)



orig_bgr = cv2.imread(str(image_input_path))
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
h0, w0   = orig_rgb.shape[:2]

img = cv2.resize(orig_rgb, IMG_SZ, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
img_input = img[None, ...]        # (1,H,W,C)  NHWC

# ----- inferência -----
pred = session.run([output_name], {input_name: img_input})[0]
print("pred shape:", pred.shape)

# ----- pós: normaliza formato e aplica sigmoid se necessário -----
def to_2d_prob(arr):
    # arr pode estar em (N,H,W,1) ou (N,1,H,W) ou (N,H,W)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:         # NHWC → (H,W)
            arr = arr[0, ..., 0]
        elif arr.shape[1] == 1:        # NCHW → (H,W)
            arr = arr[0, 0, ...]
        else:                          # (N,H,W,C) sem canal 1 → pega 1º canal
            arr = arr[0, ..., 0]
    elif arr.ndim == 3:
        arr = arr[0, ...]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Formato de saída inesperado: {arr.shape}")

    # aplica sigmoid se não parecer probabilidade
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_min < 0.0 or arr_max > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr

prob_2d = to_2d_prob(pred)                # (H,W) ∈ [0,1]
mask_2d = (prob_2d >= prob_threshold).astype(np.uint8)

# volta para o tamanho original
mask_up = cv2.resize(mask_2d, (w0, h0), interpolation=cv2.INTER_NEAREST)

# overlay opcional
overlay = orig_rgb.copy()
overlay[mask_up == 1] = [255, 0, 0]

# ----- plot -----
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(orig_rgb); plt.title("Imagem");      plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask_up, cmap="gray"); plt.title("Máscara"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(overlay);  plt.title("Overlay");     plt.axis("off")
plt.tight_layout(); plt.show()
