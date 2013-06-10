# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

def ToImageSpace(modelArr, params, out):

	assert(modelArr.shape == out.shape)

	cdef float ang = math.radians(params[3])
	cdef float angcos = math.cos(ang)
	cdef float angsin = math.sin(ang)

	rot = np.array([[math.cos(ang), - math.sin(ang)], [math.sin(ang), math.cos(ang)]])
	rotModel = np.dot(rot, modelArr.transpose()).transpose()

	scaledModel = rotModel * params[2]
	transModel = scaledModel + params[:2]

	for i in range(transModel.shape[0]):
		for j in range(transModel.shape[1]):


			out[i,j] = transModel[i,j]
	
