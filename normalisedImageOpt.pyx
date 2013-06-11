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
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, temp, pix, valid)
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
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, self.bilinearTemp, self.pixArr, self.pixValid)
		return self.pixArr

	def GetPixelImPos(self, x, y, out = None):

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
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, self.bilinearTemp, self.singlePixArr, self.singlePixValid)
		
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
		pxutil.GetPixIntensityAtLoc(self.imarr, imPos, 2, temp, pix, valid)
		return pix

	def NumPoints(self):
		return min(len(self.model), len(self.procShape))

	def GetParams(self):
		return self.params

	def NumChannels(self):
		return self.imarr.shape[2]

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
		cdef int numChan = self.normIm.NumChannels()
		cdef np.ndarray[np.float64_t, ndim=1] total = np.zeros((numChan))
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

