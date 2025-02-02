import numpy as np
def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	values = []
	rows_indices = []
	col_pointers = [0]

	dense_matrix = np.array(dense_matrix)
	for col in dense_matrix.T:

		for row_indx, value in enumerate(col):

			if value != 0:
				values.append(value)
				rows_indices.append(row_indx)
			
		col_pointers.append(len(values))
	
	return values, rows_indices, col_pointers
