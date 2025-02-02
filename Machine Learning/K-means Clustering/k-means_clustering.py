import numpy as np
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	
	for _ in range(max_iterations):

		cluster_groups = [[] for _ in range(k)]
		for i, point in enumerate(points):

			min_indx = np.argmin([np.linalg.norm(np.array(point) - np.array(center)) for center in initial_centroids])

			cluster_groups[min_indx].append(point)
		
		final_centroids = [tuple(np.mean(np.array(cluster), axis=0)) for cluster in cluster_groups]

		initial_centroids = final_centroids.copy()

	return final_centroids