import numpy as np

from .lab_samples import get_sample_image_paths


def rgb_to_grayscale_weighted(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def adaptive_threshold_minmax(gray: np.ndarray, window: int = 3) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("Размер окна должен быть нечетным и >= 3.")

    pad = window // 2
    padded = np.pad(gray, pad, mode="edge")
    h, w = gray.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            win = padded[y : y + window, x : x + window]
            min_v = int(win.min())
            max_v = int(win.max())
            threshold = (min_v + max_v) / 2.0
            result[y, x] = 255 if gray[y, x] >= threshold else 0

    return result


def fetch_sample_image_paths() -> list[str]:
    return get_sample_image_paths()
