import os
from typing import Tuple

import numpy as np
from PIL import Image

from .lab_constants import ALLOWED_EXTENSIONS


def validate_path(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Поддерживаются только PNG и BMP.")


def load_rgb_image(path: str) -> np.ndarray:
    validate_path(path)
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_rgb_image(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_gray_image(arr: np.ndarray, path: str) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def split_rgb_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return r, g, b


def rgb_channel_as_image(channel: np.ndarray, color_index: int) -> np.ndarray:
    h, w = channel.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, color_index] = channel
    return result


# =========================
# RGB -> HSI (manual)
# =========================


def rgb_to_hsi_intensity(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64)
    intensity = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
    return np.clip(np.rint(intensity), 0, 255).astype(np.uint8)


def invert_hsi_intensity(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64)
    intensity = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
    new_intensity = 255.0 - intensity

    result = np.zeros_like(rgb)
    non_zero_mask = intensity > 1e-9

    scale = np.zeros_like(intensity)
    scale[non_zero_mask] = new_intensity[non_zero_mask] / intensity[non_zero_mask]

    for c in range(3):
        result[:, :, c] = rgb[:, :, c] * scale

    zero_mask = ~non_zero_mask
    result[zero_mask] = 255.0

    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


# =========================
# Prediscretization (manual)
# =========================


def nearest_resize_manual(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Коэффициент должен быть больше 0.")

    src_h, src_w, channels = image.shape
    dst_h = max(1, int(round(src_h * scale)))
    dst_w = max(1, int(round(src_w * scale)))

    result = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)

    for y_dst in range(dst_h):
        y_src = min(src_h - 1, int(y_dst / scale))
        for x_dst in range(dst_w):
            x_src = min(src_w - 1, int(x_dst / scale))
            result[y_dst, x_dst] = image[y_src, x_src]

    return result


def stretch_manual(image: np.ndarray, m: float) -> np.ndarray:
    return nearest_resize_manual(image, m)


def compress_manual(image: np.ndarray, n: float) -> np.ndarray:
    if n <= 0:
        raise ValueError("N должно быть больше 0.")
    return nearest_resize_manual(image, 1.0 / n)


def rediscretize_two_pass(image: np.ndarray, m: float, n: float) -> np.ndarray:
    stretched = stretch_manual(image, m)
    return compress_manual(stretched, n)


def rediscretize_one_pass(image: np.ndarray, k: float) -> np.ndarray:
    return nearest_resize_manual(image, k)
