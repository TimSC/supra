# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np

class KernelFilter:
	def __init__(self, normImIn, kernelIn = None, offsetsIn = None):

		if kernelIn is not None:
			self.kernel = np.array(kernelIn, dtype=np.int32)
		else:
			self.kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)

		if offsetsIn is not None:
			self.halfw = (len(self.kernel) - 1) / 2
			self.offsets = np.array(offsetsIn, dtype=np.int32)
		else:
			self.offsets, self.halfw = CalcKernelOffsets(self.kernel)

		self.scale = 0.05
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
		cdef double sc = self.scale
	
		arr = np.array((self.offsets * sc) + (xOff, yOff))
		pixs = self.normIm.GetPixelsImPos(arr)
		total = pixs.sum(axis=0)

		if self.absVal:
			return np.abs(total)
		return total

	def GetPixelsImPos(self, pixPosLi):
		out = []
		for pos in pixPosLi:
			out.append(self.GetPixelImPos(pos[0], pos[1]))
		return out

def CalcKernelOffsets(kernel):
	cdef int hw = (kernel.shape[0] - 1) / 2

	offsets = []
	for x in range(-hw, hw+1):
		for y in range(-hw, hw+1):
			offsets.append((x,y))
	offsets = np.array(offsets, dtype = np.int32)
	return offsets, hw

def ExtractPatchAtImg(normImage, double ptX, \
	double ptY, \
	int patchw=24, \
	int patchh=24, \
	double scale=0.08):

	cdef int x, y, ch
	cdef float rawX, rawY
	cdef np.ndarray[np.uint8_t, ndim=1] tmp = np.empty(normImage.imarr.shape[2], dtype=np.uint8)

	cdef np.ndarray[np.uint8_t, ndim=3] localPatch = np.zeros((patchh, patchw, normImage.imarr.shape[2]), dtype=np.uint8)
	for x in range(patchw):
		for y in range(patchh):
			rawX = (x-((patchw-1)/2))*scale+ptX
			rawY = (y-((patchh-1)/2))*scale+ptY
			normImage.GetPixelImPos(rawX, rawY, tmp)

			for ch in range(tmp.shape[0]):
				localPatch[y,x,ch] = tmp[ch]
	return localPatch


