# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

cdef int LimitToRangeByMirroring(int val, int lower, int upper):
	cdef int valid = 0
	while not valid:
		valid = 1
		if val >= upper:
			val = 2 * upper - val - 1
			valid = 0
		if val < lower:
			val = - val
			valid = 0
	return val

cdef BilinearSample(np.ndarray[np.uint8_t, ndim=3] imgPix,
	float x, float y, 
	np.ndarray[np.float64_t, ndim=2] p, #Temporary storage
	np.ndarray[np.float64_t, ndim=2] out,
	int row):

	cdef int c
	cdef int xi = int(x)
	cdef int xi2 = xi+1
	cdef double xfrac = x - xi
	cdef int yi = int(y)
	cdef int yi2 = yi+1
	cdef double yfrac = y - yi
	cdef double c1, c2

	#Ensure position is in bounds by reflect image as necessary
	xi = LimitToRangeByMirroring(xi, 0, imgPix.shape[1])
	xi2 = LimitToRangeByMirroring(xi2, 0, imgPix.shape[1])
	yi = LimitToRangeByMirroring(yi, 0, imgPix.shape[0])
	yi2 = LimitToRangeByMirroring(yi2, 0, imgPix.shape[0])

	#Get surrounding pixels
	for c in range(imgPix.shape[2]):
		p[0,c] = imgPix[yi, xi, c]
		p[1,c] = imgPix[yi, xi2, c]
		p[2,c] = imgPix[yi2, xi, c]
		p[3,c] = imgPix[yi2, xi2, c]

	for c in range(imgPix.shape[2]):
		c1 = p[0,c] * (1.-xfrac) + p[1,c] * xfrac
		c2 = p[2,c] * (1.-xfrac) + p[3,c] * xfrac
		out[row,c] = c1 * (1.-yfrac) + c2 * yfrac

def GetPixIntensityAtLoc(np.ndarray[np.uint8_t, ndim=3] iml, \
	np.ndarray[np.float64_t, ndim=2] imLoc, \
	int randomOob, \
	np.ndarray[np.float64_t, ndim=2] temp, \
	np.ndarray[np.float64_t, ndim=2] out, \
	np.ndarray[np.int_t, ndim=1] valid):

	cdef float offsetX, offsetY
	cdef int offsetNum, oob, ch

	for offsetNum in range(imLoc.shape[0]):
		offsetX = imLoc[offsetNum, 0]
		offsetY = imLoc[offsetNum, 1]

		#Check bounds
		oob = 0
		if randomOob == 0 or randomOob == 1:
			if offsetX < 0 or offsetX >= iml.shape[1]:
				oob = 1
			if offsetY < 0 or offsetY >= iml.shape[0]:
				oob = 1

		#Get pixel at this location
		valid[offsetNum] = oob
		if not oob:
			BilinearSample(iml, offsetX, offsetY, temp, out, offsetNum)
		else:
			if randomOob == 0 or randomOob == 2:
				for ch in iml.shape[2]:
					out[offsetNum, ch] = 0
			if randomOob == 1:
				for ch in iml.shape[2]:
					out[offsetNum, ch] = np.random.uniform(0, 255)

def ITUR6012(col): #ITU-R 601-2
	return 0.299*col[0] + 0.587*col[1] + 0.114*col[2]

def ToGrey(col):
	if not hasattr(col, '__iter__'): return col
	if len(col) == 3:
		return ITUR6012(col)
	#Assumed to be already grey scale
	return col[0]
