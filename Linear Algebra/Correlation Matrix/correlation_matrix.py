import numpy as np

def calculate_correlation_matrix(X, Y=None):
	
	X_centered = X - np.mean(X, axis=0, keepdims=True)
    
	if Y is None:
		Y_centered = X_centered
	else:
		Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

	covariance_matrix = np.dot(X_centered.T, Y_centered) / (X_centered.shape[0] - 1)
	
	std_X = np.std(X, axis=0, ddof=1)
	std_Y = np.std(Y if Y is not None else X, axis=0, ddof=1)
    
	correlation_matrix = covariance_matrix / np.outer(std_X, std_Y)
    
	return correlation_matrix