# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np
import pxutil, procrustesopt, procrustes
from PIL import Image
import StringIO
import urllib2

class NormalisedImage:
	def __init__(self, urlIn, modelIn, meanFaceIn, sampleInfo):

		self.im, self.iml, self.imarr = None, None, None
		assert urlIn is not None
		if isinstance(urlIn, basestring):
			self.url = urlIn
		else:
			self.imarr = urlIn
		self.model = modelIn
		self.meanFace = meanFaceIn
		self.procShape = None
		self.params = None
		self.info = sampleInfo
		self.singlePixArr = None
	
	def LoadImage(self):

		urlImgHandle = urllib2.urlopen(self.url)
		self.im = Image.open(StringIO.StringIO(urlImgHandle.read()))
		#self.iml = self.im.load()
		self.imarr = np.array(self.im)

	def CalcProcrustes(self):
		modelArr = np.array(self.model)
		self.procShape, self.params = procrustes.CalcProcrustesOnFrame(\
			procrustes.FrameToArray(modelArr),\
			procrustes.FrameToArray(self.meanFace))

	def ClearPilImage(self):
		self.im, self.iml = None, None

	def GetPixelPos(self, ptNum, x, y):

		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		
		#Translate in normalised space, then convert back to image space
		startPoint = self.procShape[ptNum]
		modPoint = [startPoint[0]+x, startPoint[1]+y]
		modPoint2d = np.array([modPoint])

		imgPos = np.empty(modPoint2d.shape)
		procrustesopt.ToImageSpace(modPoint2d, np.array(self.params), imgPos)
		return imgPos[0]

	def GetPixelPosImPos(self, x, y):

		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		
		#Translate in normalised space, then convert back to image space
		modPoint = [x, y]
		modPoint2d = np.array([modPoint])

		imgPos = np.empty(modPoint2d.shape)
		procrustesopt.ToImageSpace(modPoint2d, np.array(self.params), imgPos)
		return imgPos[0]

	def GetPixelsPosImPos(self, normPts, out = None):

		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		if out is None:
			out = np.empty(normPts.shape)
		procrustesopt.ToImageSpace(normPts, np.array(self.params), out)
		return out

	def GetNormPos(self, x, y):
		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		return procrustes.ToProcSpace((x, y), self.params)

	def GetPixel(self, ptNum, x, y):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		imPos = self.GetPixelPos(ptNum, x, y)
		imLoc = np.array([imPos], dtype=np.float64)
		
		pix = np.empty((imLoc.shape[0], self.imarr.shape[2]))
		valid = np.empty(imLoc.shape[0], dtype=np.int)
		temp = np.empty((4, self.imarr.shape[2]))
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, 1, temp, pix, valid)
		return pix[0]

	def GetPixels(self, ptNum, pixPosLi):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		posImgLi = []
		for pos in pixPosLi:
			imPos = self.GetPixelPos(ptNum, pos[0], pos[1])
			posImgLi.append(imPos)
		imLoc = np.array(posImgLi, dtype=np.float64)
		
		self.pixArr = np.empty((imLoc.shape[0], self.imarr.shape[2]))
		self.pixValid = np.empty(imLoc.shape[0], dtype=np.int)
		self.bilinearTemp = np.empty((4, self.imarr.shape[2]))
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, 1, self.bilinearTemp, self.pixArr, self.pixValid)
		return self.pixArr

	def GetPixelImPos(self, np.ndarray[np.float64_t, ndim=2] pixPosLi, int num, out = None):

		cdef double x = pixPosLi[num,0]
		cdef double y = pixPosLi[num,1]

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		imPos = self.GetPixelPosImPos(x, y)
		imLoc = np.array([imPos], dtype=np.float64)

		if not hasattr(self, 'singlePixArr'): #Temporary function to set member var
			self.singlePixArr = None

		if self.singlePixArr is None:
			self.singlePixArr = np.empty((imLoc.shape[0], self.imarr.shape[2]))
			self.singlePixValid = np.empty(imLoc.shape[0], dtype=np.int)
			self.bilinearTemp = np.empty((4, self.imarr.shape[2]))
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, 1, self.bilinearTemp, self.singlePixArr, self.singlePixValid)
		
		px = self.singlePixArr[0]
		if out is not None:
			for ch in range(self.imarr.shape[2]):
				out[ch] = px[ch]

		return px

	def GetPixelsImPos(self, pixPosLi):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		imPos = self.GetPixelsPosImPos(pixPosLi)
		
		pix = np.empty((imPos.shape[0], self.NumChannels()))
		valid = np.empty(imPos.shape[0], dtype=np.int)
		temp = np.empty((4, self.imarr.shape[2]))
		pxutil.GetPixIntensityAtLoc(self.imarr, imPos, 2, 1, temp, pix, valid)
		return pix

	def NumPoints(self):
		return len(self.model)

	def GetParams(self):
		return self.params

	def NumChannels(self):
		return self.imarr.shape[2]

	def GetProcrustesNormedModel(self):
		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		return self.procShape

def GenSobelKernel(float angle):
	if angle == 0.:
		return [[1,0,-1],[2,0,-2],[1,0,-1]]
	if angle == 90.:
		return [[-1,-2,-1],[0,0,0],[1,2,1]]
	raise Exception("Not implemented")

cdef class KernelFilter:

	cdef np.ndarray kernel, offsets, scaleOffsets, coeffs
	cdef double scale
	cdef int halfw
	cdef normIm
	cdef int absVal, numChans

	def __init__(self, normImIn, kernelIn = None, offsetsIn = None, coeffsIn = None):

		cdef np.ndarray[np.int32_t, ndim=2] kernel = self.kernel
		cdef np.ndarray[np.int32_t, ndim=2] offsets = self.offsets
		cdef np.ndarray[np.float64_t, ndim=2] scaleOffsets = self.scaleOffsets

		if kernelIn is not None:
			kernel = np.array(kernelIn, dtype=np.int32)
		else:
			kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)
		self.kernel = kernel

		if offsetsIn is not None:
			assert coeffsIn is not None
		if offsetsIn is not None and coeffsIn is not None:
			self.halfw = (len(kernel) - 1) / 2
			offsets = np.array(offsetsIn, dtype=np.int32)
			self.coeffs = coeffsIn
		else:
			offsets, self.coeffs, self.halfw = CalcKernelOffsets(kernel)
		self.offsets = offsets

		self.scale = 0.05
		scaleOffsets = offsets * self.scale
		self.scaleOffsets = scaleOffsets

		self.normIm = normImIn
		self.absVal = True
		self.numChans = self.normIm.NumChannels()
		
	def GetPixel(self, ptNum, xOff, yOff):
		cdef np.ndarray[np.int32_t, ndim=2] kernel = self.kernel
		total = 0.

		for x in range(-self.halfw, self.halfw+1):
			for y in range(-self.halfw, self.halfw+1):
				comp = kernel[y+self.halfw][x+self.halfw]
				if comp != 0.:
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

	def GetPixelImPos(self, np.ndarray[np.float64_t, ndim=2] pixPosLi, int num):
		cdef int x, y, i
		cdef double comp

		cdef np.ndarray[np.int32_t, ndim=2] k = self.kernel
		cdef np.ndarray[np.float64_t, ndim=2] scaleOffsets = self.scaleOffsets

		cdef np.ndarray[np.float64_t, ndim=1] off = pixPosLi[num,:]
		cdef np.ndarray[np.float64_t, ndim=2] arr = scaleOffsets + off
		cdef np.ndarray[np.float64_t, ndim=2] pixs = self.normIm.GetPixelsImPos(arr)
		cdef np.ndarray[np.float32_t, ndim=1] coeffs = self.coeffs
		cdef float total = (pixs.transpose() * coeffs).sum()

		if self.absVal and total < 0.:
			return -total
		return total

	def GetPixelsImPos(self, np.ndarray[np.float64_t, ndim=2] pixPosLi):

		cdef np.ndarray[np.float64_t, ndim=2] scaleOffsets = self.scaleOffsets
		cdef np.ndarray[np.float32_t, ndim=1] coeffs = self.coeffs
		cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((pixPosLi.shape[0], self.numChans))
		cdef np.ndarray[np.float64_t, ndim=2] pos = np.empty((pixPosLi.shape[0] * scaleOffsets.shape[0], 2))

		cdef int i, j, row, ch
		for i in range(pixPosLi.shape[0]):
			for j in range(scaleOffsets.shape[0]):
				row = i * scaleOffsets.shape[0] + j
				pos[row,0] = pixPosLi[i,0] + scaleOffsets[j,0]
				pos[row,1] = pixPosLi[i,1] + scaleOffsets[j,1]

		cdef np.ndarray[np.float64_t, ndim=2] pixs = self.normIm.GetPixelsImPos(pos)
		
		for i in range(pixPosLi.shape[0]):
			for ch in range(self.numChans):
				out[i,ch] = 0.

			for j in range(scaleOffsets.shape[0]):
				row = i * scaleOffsets.shape[0] + j

				for ch in range(self.numChans):
					out[i,ch] += pixs[row, ch] * coeffs[j]
				
		return out

	def __getstate__(self):
		return ({'normIm': self.normIm, 'kernel': self.kernel, 'offsets': self.offsets, 'scaleOffsets': self.scaleOffsets, 
			'coeffs': self.coeffs, 'scale': self.scale, 'halfw': self.halfw, 'normIm': self.normIm,
			'absVal': self.absVal, 'numChans': self.numChans},)
	
	def __setstate__(self, state):
		s = state[0]
		self.kernel = s['kernel']
		self.offsets = s['offsets']
		self.scaleOffsets = s['scaleOffsets']
		self.coeffs = s['coeffs']
		self.scale = s['scale']
		self.halfw = s['halfw']
		self.normIm = s['normIm']
		self.absVal = s['absVal']
		self.numChans = s['numChans']

	def __reduce__(self):
		state = self.__getstate__()
		return (KernelFilterConstruct, state)

def KernelFilterConstruct(*args):
	kern = KernelFilter(args[0]['normIm'])
	kern.__setstate__(args)
	return kern

def CalcKernelOffsets(kernel):
	cdef int hw = (kernel.shape[0] - 1) / 2

	offsets = []
	coeffs = []
	for x in range(-hw, hw+1):
		for y in range(-hw, hw+1):
			offsets.append((x,y))
			coeffs.append(kernel[x+hw,y+hw])
	offsets = np.array(offsets, dtype = np.int32)
	coeffs = np.array(coeffs, dtype = np.float32)
	return offsets, coeffs, hw

def GenPatchOffsetList(double ptX, \
	double ptY, \
	int patchw=24, \
	int patchh=24, \
	double scale=0.08):

	cdef int x, y, count = 0
	cdef float rawX, rawY
	
	cdef np.ndarray[np.float64_t, ndim=2] imLocs = np.empty((patchw * patchh, 2), dtype=np.float64)

	for x in range(patchw):
		for y in range(patchh):
			rawX = (x-((patchw-1)/2))*scale+ptX
			rawY = (y-((patchh-1)/2))*scale+ptY
			imLocs[count,0] = rawX
			imLocs[count,1] = rawY
			count += 1

	return imLocs

def ExtractPatchAtImg(normImage, imLocs):

	cdef np.ndarray[np.uint8_t, ndim=1] tmp = np.empty(normImage.NumChannels(), dtype=np.uint8)
	pix = normImage.GetPixelsImPos(imLocs)
	return pix

