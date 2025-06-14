import os
import re


def set_cupy_cache_dir(preferred_path='D:/Temp/cupy_cache'):
    """
    设置 CuPy 缓存目录，如果当前默认路径中包含中文，则将其更改为指定的英文路径。
    """
    current_tmp = os.environ.get('TMP') or os.environ.get(
        'TEMP') or os.getenv('CUPY_CACHE_DIR', '')

    def contains_chinese(path):
        return bool(re.search(r'[\u4e00-\u9fff]', path))

    if contains_chinese(current_tmp):
        os.environ['TMP'] = preferred_path
        os.environ['TEMP'] = preferred_path
        os.environ['CUPY_CACHE_DIR'] = preferred_path
        print(
            f"⚠️  Detected Chinese characters in current cache path. CuPy cache directory has been set to: {preferred_path}")
    else:
        print(
            f"✅ No Chinese characters detected in current cache path: {current_tmp}. No change needed.")


try:
    import cupy as cp
    cp.array([1])
    xp = cp
    use_gpu = True
    print("✅ CuPy (GPU) is available, using it as backend by default.")
    set_cupy_cache_dir()
except Exception:
    import numpy as np
    xp = np
    use_gpu = False
    print("⚠️  CuPy (GPU) is not available, using NumPy (CPU) instead.")


def set_backend(backend):
    global xp, use_gpu
    if backend == 'gpu':
        if use_gpu:
            return
        try:
            import cupy as cp
            cp.array([1])
            xp = cp
            use_gpu = True
            print("✅ Switched to CuPy (GPU) backend")
            set_cupy_cache_dir()
        except Exception:
            raise ImportError("CuPy is not installed or GPU is not available.")
    elif backend == 'cpu':
        if not use_gpu:
            return
        import numpy as np
        xp = np
        use_gpu = False
        print("⚠️  Switched to NumPy (CPU) backend")
    else:
        raise ValueError("Unsupported backend. Use 'cpu' or 'gpu'.")
