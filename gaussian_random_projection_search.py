import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f}s")
        return result
    return wrapper

def load_raw_binary(path, dtype, shape):
    """
    Load a raw binary file as a 3D NumPy array
    shape = (frames, height, width)
    Usage: data_fp32 = load_raw_binary(input_file, dtype=np.float32, shape=(500, 500, 100))
    """
    try:
        data = np.fromfile(path, dtype=dtype)
        data = data.reshape(shape)
        return data
    except Exception as e:
        print(f"[Error] Cannot read raw binary file {path}: {e}")
        return None

@timeit
def euclidean_distance(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.
    """
    v1 = np.asarray(vec1).ravel()
    v2 = np.asarray(vec2).ravel()

    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes must match, got {v1.shape} and {v2.shape}")

    return np.linalg.norm(v1 - v2)

@timeit
def gaussian_random_projection_fp16(X, n_components, random_state=42, feature_block_size=50_000):
    """
    Manual Gaussian Random Projection in FP32 with feature-wise chunking.

    This avoids constructing a full (n_features, n_components) matrix in memory.
    """
    X = np.asarray(X, dtype=np.float16)
    n_samples, n_features = X.shape

    if n_components <= 0:
        raise ValueError(f"n_components must be > 0, got {n_components}")
    if feature_block_size <= 0:
        raise ValueError(f"feature_block_size must be > 0, got {feature_block_size}")
    feature_block_size = min(feature_block_size, n_features)

    rng = np.random.default_rng(random_state)
    projected = np.zeros((n_samples, n_components), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(n_components))

    for start in range(0, n_features, feature_block_size):
        end = min(start + feature_block_size, n_features)
        block_width = end - start

        random_block = rng.standard_normal(
            size=(block_width, n_components),
            dtype=np.float32,
        )
        random_block *= scale

        projected += X[:, start:end].astype(np.float32) @ random_block

    return projected
    
def main():
    # Load the raw binary data
    Data_fp16 = []
    for ts in range(1, 10):  # Assuming we have 10 time steps
        input_file = f"CLOUDf0{ts}.bin" 
        raw = load_raw_binary(input_file, dtype=np.float32, shape=(500, 500, 100))
        if raw is None:
            return
        data_fp16 = raw.astype(np.float16)
        
        data_fp16_s = data_fp16.reshape(25_000_000)
        # data_fp16_s = data_fp16_s[50*250_000:100*250_000]  # Use only the 50th block of 250,000 elements for testing
        Data_fp16.append(data_fp16_s)
    
    Data_fp16 = np.array(Data_fp16)
    print(f"Loaded data shape: {Data_fp16.shape}")

    # Determine the minimum number of components for Gaussian Random Projection
    eps = 0.1  # Set the desired distortion level
    n_samples = Data_fp16.shape[0]
    min_components = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
    
    print(f"Minimum number of components for Gaussian Random Projection: {min_components}")

    # Manual Gaussian Random Projection in FP16
    projected_data = gaussian_random_projection_fp16(
        Data_fp16,
        n_components=min_components,
        random_state=42,
        feature_block_size=50_000,
    )
    print(f"Projected data shape: {projected_data.shape}")

    for ts in range(1, 9):
        print(f"Time step {ts}:")
        print(euclidean_distance(Data_fp16[0], Data_fp16[ts]))
        print(euclidean_distance(projected_data[0], projected_data[ts]))

    # print(f"Original shape: {Data_fp16.shape}")
    # print(f"Projected shape: {projected_data.shape}")

if __name__ == "__main__":
    main()