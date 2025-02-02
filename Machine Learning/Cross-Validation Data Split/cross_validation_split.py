import numpy as np
def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
	np.random.seed(seed)
	
	indices = np.arange(len(data))
	np.random.shuffle(indices)

	data = data[indices]

	folds = np.array_split(np.arange(len(data)), k)
	splits = []

	for i in range(k):
		val_indices = folds[i]
		val_data = data[val_indices]

		train_data = [data[folds[j]] for j in range(k) if j != i]
		splits.append([train_data, val_data])

	return splits