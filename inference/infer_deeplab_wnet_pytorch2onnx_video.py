import onnxruntime as ort
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# ------------------- CONFIG -------------------

video_path  = r"inference/video/ufsc_video.mp4"

model_path = r"weights/Sky_Unet_DeepLab_v1_MobileNetv2_infer.onnx"


IMG_SZ = (320, 320)
PROB_THRESHOLD = 0.5

print(f"Modelo: {model_path}")
def preprocess_frame(frame_bgr, img_size=(320, 320)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, img_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)


def postprocess_mask(output, original_size, threshold=0.5):
    prob = output.squeeze()
    if prob.min() < 0.0 or prob.max() > 1.0:
        prob = 1.0 / (1.0 + np.exp(-prob))

    mask = (prob > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)


def infer_video(onnx_path, video_path, img_size=(320, 320), threshold=0.5):
    onnx_path = Path(onnx_path)
    video_path = Path(video_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Modelo ONNX nao encontrado: {onnx_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nao consegui abrir o video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    winname = "Original  |  Mascara Binaria"

    try:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, 1200, 400)
    except cv2.error as ex:
        cap.release()
        raise RuntimeError(
            "OpenCV sem suporte de GUI (provavel opencv-python-headless). "
            "Instale opencv-python para usar namedWindow/imshow."
        ) from ex

    for _ in tqdm(range(total_frames), desc="Processando"):
        ok, frame = cap.read()
        if not ok:
            break

        input_tensor = preprocess_frame(frame, img_size)
        output = session.run([output_name], {input_name: input_tensor})[0]

        h, w = frame.shape[:2]
        mask = postprocess_mask(output, (w, h), threshold)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        side_by_side = cv2.hconcat([frame, mask_bgr])
        cv2.imshow(winname, side_by_side)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


infer_video(model_path, video_path, IMG_SZ, PROB_THRESHOLD)
