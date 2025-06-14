try:
    import cupy as cp
    cp.array([1])
    xp = cp
    use_gpu = True
    print("✅ CuPy (GPU) is available, using it as backend")
except Exception:
    import numpy as np
    xp = np
    use_gpu = False
    print("⚠️ CuPy (GPU) is not available, using NumPy (CPU) instead")


def set_backend(backend):
    global xp, use_gpu
    if backend == 'gpu':
        try:
            import cupy as cp
            cp.array([1])
            xp = cp
            use_gpu = True
            print("✅ Switched to CuPy (GPU) backend")
        except Exception:
            raise ImportError("CuPy is not installed or GPU is not available.")
    elif backend == 'cpu':
        import numpy as np
        xp = np
        use_gpu = False
        print("⚠️ Switched to NumPy (CPU) backend")
    else:
        raise ValueError("Unsupported backend. Use 'cpu' or 'gpu'.")
