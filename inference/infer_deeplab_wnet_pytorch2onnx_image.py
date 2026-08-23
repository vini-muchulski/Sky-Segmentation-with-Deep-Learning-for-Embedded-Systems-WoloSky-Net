import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps

# ----- caminhos -----
path_folder = r"inference/imgs"

#model_path = r"weights/Sky_Unet_DeepLab_v1_MobileNetv2_infer.onnx"
model_path = r"weights/Sky_Unet_DeepLab_v2_Resnet34_infer.onnx"

# ----- parametros -----
IMG_SZ = (320, 320)
PROB_THRESHOLD = 0.5

print(f"Modelo: {model_path}")
print(f"Pasta de imagens: {path_folder}")

def load_image(image_path):
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def to_2d_prob(arr):
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[0, ..., 0]     # NHWC
        elif arr.shape[1] == 1:
            arr = arr[0, 0, ...]     # NCHW
        else:
            arr = arr[0, ..., 0]
    elif arr.ndim == 3:
        arr = arr[0, ...]
    elif arr.ndim != 2:
        raise ValueError(f"Formato de saida inesperado: {arr.shape}")

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_min < 0.0 or arr_max > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr


def build_input_tensor(img_rgb, img_size, input_shape):
    img = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # Detecta layout esperado pelo modelo:
    # [N, C, H, W] -> NCHW, [N, H, W, C] -> NHWC
    if len(input_shape) == 4 and input_shape[1] == 3:
        return np.transpose(img, (2, 0, 1))[None, ...]
    if len(input_shape) == 4 and input_shape[-1] == 3:
        return img[None, ...]

    # fallback para modelos com shape dinamico
    return np.transpose(img, (2, 0, 1))[None, ...]


def choose_output_name(outputs):
    if len(outputs) == 1:
        return outputs[0].name

    for out in outputs:
        name = (out.name or "").lower()
        if "output_fused" in name or "fused" in name:
            return out.name
    return outputs[0].name


def infer_folder(onnx_path, image_folder):
    onnx_path = Path(onnx_path)
    image_folder = Path(image_folder)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Modelo ONNX nao encontrado: {onnx_path}")
    if not image_folder.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {image_folder}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".jfif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"Nenhuma imagem encontrada em {image_folder}")
        return

    print(f"Encontradas {len(image_files)} imagens")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    input_name = input_meta.name
    input_shape = input_meta.shape
    output_name = choose_output_name(output_meta)

    print("input:", input_shape)
    print("output(s):", [(o.name, o.shape) for o in output_meta])
    print("usando saida:", output_name)

    for idx, image_input_path in enumerate(image_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Processando imagem {idx}/{len(image_files)}: {image_input_path.name}")
        print(f"{'=' * 60}")

        try:
            orig_rgb = load_image(image_input_path)
            h0, w0 = orig_rgb.shape[:2]
            print(f"Dimensoes originais: {w0}x{h0}")

            input_tensor = build_input_tensor(orig_rgb, IMG_SZ, input_shape)
            pred = session.run([output_name], {input_name: input_tensor})[0]
            print("pred shape:", pred.shape)

            prob_2d = to_2d_prob(pred)
            mask_2d = (prob_2d >= PROB_THRESHOLD).astype(np.uint8)
            mask_up = cv2.resize(mask_2d, (w0, h0), interpolation=cv2.INTER_NEAREST)

            overlay = orig_rgb.copy()
            overlay[mask_up == 1] = [255, 0, 0]

            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Imagem {idx}/{len(image_files)}: {image_input_path.name}", fontsize=14)

            plt.subplot(1, 3, 1)
            plt.imshow(orig_rgb)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(mask_up, cmap="gray")
            plt.title("Mascara")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            print(f"Imagem {idx} processada com sucesso")
        except Exception as ex:
            print(f"Erro ao processar {image_input_path.name}: {str(ex)}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Processamento concluido! Total de imagens: {len(image_files)}")
    print(f"{'=' * 60}")


infer_folder(model_path, path_folder)
