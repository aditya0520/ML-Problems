import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
	m, _ = data.shape
	mean = np.mean(data, axis = 0, keepdims=True)
	std = np.std(data, axis = 0, keepdims=True)
	standardized_data = (data - mean) / std

	covariance = standardized_data.T @ standardized_data / (m - 1)

	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	max_k_indx = np.argsort(eigenvalues)[::-1][:k]

	principal_components = eigenvectors[:, max_k_indx]


	return np.round(principal_components, 4)