import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	values = []
	columns_indices = []
	row_pointers = [0]

	for row in dense_matrix:

		for col_indx, value in enumerate(row):

			if value != 0:
				values.append(value)
				columns_indices.append(col_indx)
			
		row_pointers.append(len(values))
	
	return values, columns_indices, row_pointers

