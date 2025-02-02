import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

	x = [0] * A.shape[0]
	temp_x = x.copy()

	for i in range(n):

		for j in range(len(x)):
			mask = np.ones(b.shape[0])
			mask[j] = 0
			x[j] = (b[j] - np.sum(A[j] * temp_x * mask)) / A[j, j]

		temp_x = x.copy()

	return x