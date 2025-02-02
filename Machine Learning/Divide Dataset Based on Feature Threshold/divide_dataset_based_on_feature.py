import numpy as np

def divide_on_feature(X, feature_i, threshold):

	true_condition = X[:, feature_i] >= threshold

	true_indices = np.where(true_condition)
	false_indices = np.where(~true_condition)
	
	return X[true_indices], X[false_indices]
	