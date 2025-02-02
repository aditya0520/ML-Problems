import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	
	x = [0] * A.shape[0] if x_ini is None else x_ini

	for i in range(n):

		for j in range(len(x)):
			mask = np.ones(b.shape[0])
			mask[j] = 0
			x[j] = (b[j] - np.sum(A[j] * x * mask)) / A[j, j]

	return x
