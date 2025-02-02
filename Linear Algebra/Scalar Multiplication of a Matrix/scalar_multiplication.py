import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

	matrix = np.array(matrix)
	result = scalar * matrix
	return list(result)