# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

class KernelFilter:
	def __init__(self, normImIn):
		self.kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)
		self.offsets = None
		self.scale = 0.05
		self.halfw = (len(self.kernel) - 1) / 2
		self.normIm = normImIn
		self.absVal = True
		
	def GetPixel(self, ptNum, xOff, yOff):
		total = 0.
		for x in range(-self.halfw, self.halfw+1):
			for y in range(-self.halfw, self.halfw+1):
				comp = self.kernel[y+self.halfw][x+self.halfw]
				total += self.normIm.GetPixel(ptNum, self.scale*x+xOff, self.scale*y+yOff) * comp
		#print xOff, yOff, total
		if self.absVal:
			return np.abs(total)
		return total

	def GetPixels(self, ptNum, pixPosLi):
		out = []
		for pos in pixPosLi:
			out.append(self.GetPixel(ptNum, pos[0], pos[1]))
		return out

	def GetPixelImPos(self, double xOff, double yOff):
		cdef int x, y, i
		cdef double comp
		cdef np.ndarray[np.float64_t, ndim=1] total = np.zeros((self.normIm.imarr.shape[2]))
		cdef np.ndarray[np.int32_t, ndim=2] k = self.kernel
		cdef np.ndarray[np.int32_t, ndim=2] offsetArr
		cdef int hw = self.halfw
		cdef double sc = self.scale
		
		if self.offsets is None:
			self.offsets = []
			for x in range(-hw, hw+1):
				for y in range(-hw, hw+1):
					self.offsets.append((x,y))
			self.offsets = np.array(self.offsets, dtype = np.int32)
		
		pixs = self.normIm.GetPixelsImPos(self.offsets)
		total = pixs.sum(axis=0)

		#for i in range(offsetArr.shape[0]):
		#	comp = k[offsetArr[i,1]+hw, offsetArr[i,0]+hw]
		#	xx = self.normIm.GetPixelImPos(sc*offsetArr[i,0]+xOff, sc*offsetArr[i,1]+yOff)
		#	total += xx * comp

		#print xOff, yOff, total
		if self.absVal:
			return np.abs(total)
		return total

	def GetPixelsImPos(self, pixPosLi):
		out = []
		for pos in pixPosLi:
			out.append(self.GetPixelImPos(pos[0], pos[1]))
		return out


