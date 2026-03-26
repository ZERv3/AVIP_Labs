from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImageState:
    path: Optional[str] = None
    image: Optional[np.ndarray] = None  # uint8 RGB, shape (H, W, 3)
    processed: Optional[np.ndarray] = None  # uint8 RGB or Gray (H, W, 3) or (H, W)
