import argparse
import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection, GaussianRandomProjection
import time
from functools import wraps

try:
    import torch
except ImportError:
    torch = None

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
def gaussian_random_projection_fp16_cpu(X, n_components, random_state=42, feature_block_size=50_000):
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

@timeit
def gaussian_random_projection_fp16_gpu(X, n_components, random_state=42, feature_block_size=50_000):
    """
    Gaussian Random Projection on CUDA using PyTorch.

    Uses fp16 for inputs/random matrix multiplication and fp32 for accumulation.
    Returns a NumPy array on CPU with dtype float32.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Install torch with CUDA support to use this function.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Ensure a CUDA-capable GPU and CUDA-enabled PyTorch are installed.")

    X = np.asarray(X, dtype=np.float16)
    n_samples, n_features = X.shape

    if n_components <= 0:
        raise ValueError(f"n_components must be > 0, got {n_components}")
    if feature_block_size <= 0:
        raise ValueError(f"feature_block_size must be > 0, got {feature_block_size}")
    feature_block_size = min(feature_block_size, n_features)

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)

    X_gpu = torch.from_numpy(X).to(device=device, dtype=torch.float16, non_blocking=True)
    projected_gpu = torch.zeros((n_samples, n_components), device=device, dtype=torch.float32)
    scale = np.float32(1.0 / np.sqrt(n_components))

    for start in range(0, n_features, feature_block_size):
        end = min(start + feature_block_size, n_features)
        block_width = end - start

        random_block = torch.randn(
            (block_width, n_components),
            device=device,
            dtype=torch.float16,
            generator=generator,
        )
        random_block = random_block * scale

        x_block = X_gpu[:, start:end]
        block_out = torch.matmul(x_block, random_block).to(torch.float32)
        projected_gpu += block_out

    return projected_gpu.cpu().numpy()
    
def main():
    parser = argparse.ArgumentParser(description="Compare pairwise distances before/after random projection.")
    parser.add_argument(
        "--projection-method",
        choices=["manual_cpu", "manual_cuda", "sklearn_sparse", "sklearn_gaussian"],
        default="manual_cuda",
        help="Projection backend to use",
    )
    args = parser.parse_args()

    data_dir = "/home/jwang96/datasets/Hurricane_new/clean-data-Jinyang/"
    field_name = ["CLOUD", "P", "PRECIP", "QCLOUD", "QGRAUP", "QICE", "QRAIN", "QSNOW", "QVAPOR", "TC", "U", "V", "W"]
    projection_method = args.projection_method

    timestep = 24
    data_shape = (500, 500, 100)

    field_vectors = {}
    for field in field_name:
        input_file = f"{data_dir}/{field}/{field}f{timestep:02d}.bin"
        raw = load_raw_binary(input_file, dtype=np.float32, shape=data_shape)
        if raw is None:
            return
        field_vectors[field] = raw.reshape(-1)

    print(f"Loaded timestep f{timestep:02d} for {len(field_vectors)} fields")

    n_samples = len(field_name)
    eps = 0.1
    min_components = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
    print(f"Minimum number of components for Random Projection: {min_components}")

    # performing Random Projection using the selected method
    data_matrix = np.stack([field_vectors[field] for field in field_name], axis=0)
    if projection_method == "manual_cuda":
        if torch is None or not torch.cuda.is_available():
            print("CUDA backend unavailable; falling back to manual CPU projection")
            projected_matrix = gaussian_random_projection_fp16(
                data_matrix,
                n_components=min_components,
                random_state=42,
                feature_block_size=50_000,
            )
            print("Projection backend: manual_cpu")
        else:
            projected_matrix = gaussian_random_projection_fp16_cuda(
                data_matrix,
                n_components=min_components,
                random_state=42,
                feature_block_size=50_000,
            )
            print("Projection backend: manual_cuda")
    elif projection_method == "manual_cpu":
        projected_matrix = gaussian_random_projection_fp16(
            data_matrix,
            n_components=min_components,
            random_state=42,
            feature_block_size=50_000,
        )
        print("Projection backend: manual_cpu")
    elif projection_method == "sklearn_sparse":
        transformer = SparseRandomProjection(n_components=min_components, random_state=42)
        projected_matrix = transformer.fit_transform(data_matrix).astype(np.float32)
        print("Projection backend: sklearn_sparse")
    elif projection_method == "sklearn_gaussian":
        transformer = GaussianRandomProjection(n_components=min_components, random_state=42)
        projected_matrix = transformer.fit_transform(data_matrix).astype(np.float32)
        print("Projection backend: sklearn_gaussian")
    else:
        raise ValueError(
            f"Unknown projection_method='{projection_method}'. "
            "Choose from: manual_cpu, manual_cuda, sklearn_sparse, sklearn_gaussian"
        )

    projected_vectors = {
        field: projected_matrix[idx]
        for idx, field in enumerate(field_name)
    }

    print("\nPairwise Euclidean distances:")
    for i, field_i in enumerate(field_name):
        for j, field_j in enumerate(field_name):
            dist_orig = euclidean_distance(field_vectors[field_i], field_vectors[field_j])
            print(f"Original distance: {field_i:>8} vs {field_j:<8}: {dist_orig:.6e}")
            dist_proj = euclidean_distance(projected_vectors[field_i], projected_vectors[field_j])
            print(f"Projected distance: {field_i:>8} vs {field_j:<8}: {dist_proj:.6e}")

if __name__ == "__main__":
    main()