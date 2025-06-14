try:
    import cupy as cp
    cp.array([1])
    xp = cp
    use_gpu = True
    print("✅ Using CuPy (GPU)")
except Exception:
    import numpy as np
    xp = np
    use_gpu = False
    print("⚠️ Using NumPy (CPU)")
