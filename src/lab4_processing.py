import numpy as np

from .lab_samples import get_sample_image_paths

KAYALI_GX = np.array(
    [
        [6, 0, -6],
        [0, 0, 0],
        [-6, 0, 6],
    ],
    dtype=np.float32,
)

KAYALI_GY = np.array(
    [
        [-6, 0, 6],
        [0, 0, 0],
        [6, 0, -6],
    ],
    dtype=np.float32,
)


def convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Ожидается 2D массив (полутон).")
    if kernel.shape != (3, 3):
        raise ValueError("Ожидается ядро 3x3.")

    pad = 1
    padded = np.pad(gray.astype(np.float32), pad, mode="edge")
    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            window = padded[y : y + 3, x : x + 3]
            result[y, x] = float(np.sum(window * kernel))

    return result


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_v) / (max_v - min_v)
    return np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)


def kayali_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = convolve2d(gray, KAYALI_GX)
    gy = convolve2d(gray, KAYALI_GY)
    g = np.sqrt(gx * gx + gy * gy)

    gx_n = normalize_to_uint8(gx)
    gy_n = normalize_to_uint8(gy)
    g_n = normalize_to_uint8(g)

    return gx_n, gy_n, g_n


def fetch_sample_image_paths() -> list[str]:
    return get_sample_image_paths()
