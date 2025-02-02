import numpy as np

def make_diagonal(x):
	x = np.diag(x, k=0)
	return x