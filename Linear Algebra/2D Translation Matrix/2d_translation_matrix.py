import numpy as np
def translate_object(points, tx, ty):
	
	points = np.array(points)
	shift = np.array([tx,ty])
	np.expand_dims(shift, 0)
	translated_points = points + shift
	return translated_points
