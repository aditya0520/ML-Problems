import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	matrix = np.array(matrix)
	trace = -np.sum(matrix * np.eye(len(matrix), dtype=bool))
	det = np.linalg.det(matrix)

	eig1 = (-trace + np.sqrt(trace ** 2 - 4 * det)) / 2
	eig2 = (-trace - np.sqrt(trace ** 2 - 4 * det)) / 2
	return [eig1,eig2]