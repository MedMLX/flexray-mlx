"""NumPy inputs shared with independent upstream reference producers."""

import numpy as np
from numpy.typing import NDArray

type HostArray = NDArray[np.generic]
type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
