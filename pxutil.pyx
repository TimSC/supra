# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

cdef int BilinearSample(np.ndarray[np.uint8_t, ndim=3] imgPix,
	float x, float y, 
	np.ndarray[np.float64_t, ndim=2] p, #Temporary storage
	np.ndarray[np.float64_t, ndim=2] out,
	int row):

	cdef int c
	cdef int xi = int(x)
	cdef double xfrac = x - xi
	cdef int yi = int(y)
	cdef double yfrac = y - yi
	cdef double c1, c2

	#Check bounds
	if xi < 0 or xi + 1 >= imgPix.shape[1]:
		return 0
	if yi < 0 or yi + 1 >= imgPix.shape[0]:
		return 0

	#Get surrounding pixels
	for c in range(imgPix.shape[2]):
		p[0,c] = imgPix[yi, xi, c]
		p[1,c] = imgPix[yi, xi+1, c]
		p[2,c] = imgPix[yi+1, xi, c]
		p[3,c] = imgPix[yi+1, xi+1, c]

	for c in range(imgPix.shape[2]):
		c1 = p[0,c] * (1.-xfrac) + p[1,c] * xfrac
		c2 = p[2,c] * (1.-xfrac) + p[3,c] * xfrac
		out[row,c] = c1 * (1.-yfrac) + c2 * yfrac

	return 1

def GetPixIntensityAtLoc(np.ndarray[np.uint8_t, ndim=3] iml, 
	np.ndarray[np.float64_t, ndim=2] imLoc, int randomOob):

	cdef np.ndarray[np.float64_t, ndim=2] out = np.zeros((imLoc.shape[0], iml.shape[2]))
	cdef np.ndarray[np.int_t, ndim=1] valid = np.zeros(imLoc.shape[0], dtype=np.int)
	cdef double x, y
	cdef float offsetX, offsetY
	cdef int offsetNum

	if not randomOob:
		out = np.zeros((imLoc.shape[0], iml.shape[2]))
		valid = np.zeros(imLoc.shape[0], dtype=np.int)
	else:
		out = np.array(np.random.random_integers(0, 255, size=(imLoc.shape[0], iml.shape[2])), dtype=np.float)
		valid = np.zeros(imLoc.shape[0], dtype=np.int)

	cdef np.ndarray[np.float64_t, ndim=2] temp = np.empty((4, iml.shape[2]))

	for offsetNum in range(imLoc.shape[0]):
		offsetX = imLoc[offsetNum, 0]
		offsetY = imLoc[offsetNum, 1]

		#Get pixel at this location
		valid[offsetNum] = BilinearSample(iml, offsetX, offsetY, temp, out, offsetNum)
	return out, valid

def ITUR6012(col): #ITU-R 601-2
	return 0.299*col[0] + 0.587*col[1] + 0.114*col[2]

def ToGrey(col):
	if not hasattr(col, '__iter__'): return col
	if len(col) == 3:
		return ITUR6012(col)
	#Assumed to be already grey scale
	return col[0]
