import numpy as np

def rref(matrix):
	
	matrix = matrix.astype(np.float32)
	rows, cols = matrix.shape
	pivot_row = 0

	for pivot_col in range(cols):

		max_row = np.argmax(np.abs(matrix[pivot_row:rows, pivot_col])) + pivot_row

		if matrix[max_row, pivot_col] == 0:
			continue
		
		matrix[[pivot_row, max_row]] = matrix[[max_row, pivot_row]]

		matrix[pivot_row] = matrix[pivot_row] / matrix[pivot_row, pivot_col]

		for row in range(rows):
			if row != pivot_row:
				matrix[row] -= matrix[row, pivot_col] * matrix[pivot_row]
		
		pivot_row += 1

		if pivot_row == rows:
			break
	
	return matrix

