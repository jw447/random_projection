import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection
from sklearn.datasets import make_classification

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

def euclidean_distance(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.
    """
    v1 = np.asarray(vec1).ravel()
    v2 = np.asarray(vec2).ravel()

    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes must match, got {v1.shape} and {v2.shape}")

    return np.linalg.norm(v1 - v2)
    
def main():
    # Load the raw binary data
    Data_fp32 = []
    for ts in [1, 2]:
        input_file = f"CLOUDf0{ts}.bin" 
        data_fp32 = load_raw_binary(input_file, dtype=np.float32, shape=(500, 500, 100))
        if data_fp32 is None:
            return
        
        data_fp32_s = data_fp32.reshape(25000000)
        Data_fp32.append(data_fp32_s)
    
    Data_fp32 = np.array(Data_fp32)
    print(f"Loaded data shape: {Data_fp32.shape}")

    # Determine the minimum number of components for Gaussian Random Projection
    eps = 0.1  # Set the desired distortion level
    n_samples = 2
    min_components = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
    
    print(f"Minimum number of components for Gaussian Random Projection: {min_components}")
    
    # Create a Gaussian Random Projection transformer
    transformer = GaussianRandomProjection(n_components=min_components)
    
    # Fit and transform the data
    projected_data = transformer.fit_transform(Data_fp32)
    print(f"Projected data shape: {projected_data.shape}")

    print(euclidean_distance(Data_fp32[0], Data_fp32[1]))
    print(euclidean_distance(Data_fp32[0], Data_fp32[2]))

    print(euclidean_distance(projected_data[0], projected_data[1]))
    print(euclidean_distance(projected_data[0], projected_data[2]))

    print(f"Original shape: {Data_fp32.shape}")
    print(f"Projected shape: {projected_data.shape}")

if __name__ == "__main__":
    main()