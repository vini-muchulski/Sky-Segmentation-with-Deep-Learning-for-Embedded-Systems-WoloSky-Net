import cv2
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

# ------------------- CONFIG -------------------
video_input_path = r"video/ufsc_video.mp4"

model_path = r"weights/v2_Wolosky_U2Net_Sam3_320x320_infer.onnx"

#IMG_SZ = (320, 320)
IMG_SZ = (224, 224)
#IMG_SZ = (128, 128)

prob_threshold = 0.5


# ---------------- ONNX Runtime ----------------
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
available_providers = ort.get_available_providers()
providers = (  ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"])
sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
print(f"ONNX Runtime providers: {sess.get_providers()}")
input_id = sess.get_inputs()[0].name
output_id = sess.get_outputs()[0].name  # modelo com saida unica (d0)


# ---------------- Funcoes utilitarias ----------
def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SZ).astype(np.float32) / 255.0
    return img[None]  # (1,H,W,C)


def logits_to_mask(logits):
    """Logits -> probabilidade -> mascara 0/1 em float32."""
    if logits.ndim == 4:
        logits = logits[0, ..., 0]
    if logits.min() < 0 or logits.max() > 1:
        logits = 1.0 / (1.0 + np.exp(-logits))
    return (logits >= prob_threshold).astype(np.float32)


# ---------------- Video ------------------------
cap = cv2.VideoCapture(video_input_path)
assert cap.isOpened(), f"Nao consegui abrir {video_input_path}"

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
winname = "Original  |  Mascara (ceu=branco)"

try:
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 1200, 400)
except cv2.error as ex:
    raise RuntimeError(
        "OpenCV sem suporte de GUI (provavel opencv-python-headless). "
        "Instale opencv-python para usar namedWindow/imshow."
    ) from ex


for _ in tqdm(range(total_frames), desc="Processando"):
    ok, frame = cap.read()
    if not ok:
        break

    # ---------- inferencia ----------
    inp = preprocess(frame)
    logits = sess.run([output_id], {input_id: inp})[0]
    mask = logits_to_mask(logits)

    # ---------- preparar visualizacao ----------
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    side_by_side = cv2.hconcat([frame, mask_vis])
    cv2.imshow(winname, side_by_side)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC ou 'q'
        break

cap.release()
cv2.destroyAllWindows()
