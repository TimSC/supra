# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np
from libc.math cimport sin, cos

def ToImageSpace(np.ndarray[np.double_t, ndim=2] modelArr, \
	np.ndarray[np.double_t, ndim=1] params, \
	np.ndarray[np.double_t, ndim=2] out):

	assert(modelArr.shape[0] == out.shape[0])
	assert(modelArr.shape[1] == out.shape[1])

	cdef float ang = params[3] * 3.14159265 / 180.
	cdef float angcos = cos(ang)
	cdef float angsin = sin(ang)

	cdef int i

	for i in range(modelArr.shape[0]):
		out[i,0] = (modelArr[i,0] * angcos - modelArr[i,1] * angsin) * params[2] + params[0]
		out[i,1] = (modelArr[i,0] * angsin + modelArr[i,1] * angcos) * params[2] + params[1]


