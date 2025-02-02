import numpy as np
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    
	G = A.T @ A
	m, n = A.shape

	V = np.eye(n)

	for _ in range(1):

		diagonal_mask = np.eye(m,n, dtype=bool)

		off_diagonal = G[~diagonal_mask]
		flat_index = np.argmax(off_diagonal)

		row, col = np.argwhere(~diagonal_mask)[flat_index]

		theta = 0.5 * np.arctan2(2 * G[row, col], G[row, row] - G[col, col])

		c, s = np.cos(theta), np.sin(theta)

		R = np.eye(n)

		R[row, row] = R[col, col] = c
		R[row, col] = -s
		R[col, row] = s
		
		G = R.T @ G @ R
		V = V @ R
	
	singular_values = np.sqrt(np.diag(G))
	Sigma = np.zeros((m, n))
	np.fill_diagonal(Sigma, singular_values)

	U = A @ V @ np.linalg.pinv(Sigma)

	return U, singular_values, V.T
