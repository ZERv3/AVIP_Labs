import numpy as np

from .lab_samples import get_sample_image_paths


def max_filter_2d(image: np.ndarray, window: int = 3) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("Размер окна должен быть нечетным и >= 3.")
    if image.ndim != 2:
        raise ValueError("Ожидается 2D массив (полутон/монохром).")

    pad = window // 2
    padded = np.pad(image, pad, mode="edge")
    h, w = image.shape
    result = np.zeros((h, w), dtype=image.dtype)

    for y in range(h):
        for x in range(w):
            result[y, x] = padded[y : y + window, x : x + window].max()

    return result


def fringe_erase_black(image: np.ndarray, window: int = 3) -> np.ndarray:
    return max_filter_2d(image, window=window)


def diff_xor(binary_a: np.ndarray, binary_b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(binary_a, binary_b).astype(np.uint8)


def diff_abs(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    return np.abs(gray_a.astype(np.int16) - gray_b.astype(np.int16)).astype(np.uint8)


def boost_diff(diff: np.ndarray, factor: float = 10.0) -> np.ndarray:
    boosted = np.clip(diff.astype(np.float32) * factor, 0, 255)
    return boosted.astype(np.uint8)


def fetch_sample_image_paths() -> list[str]:
    return get_sample_image_paths()
