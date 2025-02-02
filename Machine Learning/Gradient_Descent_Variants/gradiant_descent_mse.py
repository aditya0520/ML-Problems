import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):

	m, n = X.shape

	if method == 'batch':
		for _ in range(n_iterations):
			
			gradient = 2 / m * X.T @ (X @ weights - y)
			weights -= learning_rate * gradient
	
	elif method == 'stochastic':

		for _ in range(n_iterations):
			for i, x in enumerate(X):
				x_i = X[i, :].reshape(1, -1)
				y_i = y[i]
				gradient = 2 * x_i.T @ (x_i @ weights - y_i)
				weights -= learning_rate * gradient
	
	elif method == 'mini_batch':
		
		for _ in range(n_iterations):

			for start in range(0, m, batch_size):
				end = start + batch_size
				X_batch = X[start:end] 
				y_batch = y[start:end]  
				
				gradient = 2 / X_batch.shape[0] * X_batch.T @ (X_batch @ weights - y_batch)
				weights -= learning_rate * gradient
	
	return weights
