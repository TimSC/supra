
import urllib2
import StringIO, procrustes, pxutil, math
from PIL import Image
import numpy as np

class NormalisedImage:
	def __init__(self, urlIn, modelIn, meanFaceIn, sampleInfo):
		self.url = urlIn
		self.model = modelIn
		self.meanFace = meanFaceIn
		self.im, self.iml, self.imarr = None, None, None
		self.procShape = None
		self.params = None
		self.info = sampleInfo
	
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
		imgPos = procrustes.ToImageSpace(np.array([modPoint]), self.params)
		return imgPos[0]

	def GetPixelPosImPos(self, x, y):

		#Lazy procrustes calculation
		if self.params is None:
			self.CalcProcrustes()
		
		#Translate in normalised space, then convert back to image space
		modPoint = [x, y]
		imgPos = procrustes.ToImageSpace(np.array([modPoint]), self.params)
		return imgPos[0]

	def GetPixel(self, ptNum, x, y):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		imPos = self.GetPixelPos(ptNum, x, y)
		imLoc = np.array([imPos], dtype=np.float64)
		
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 1)[0][0]

	def GetPixels(self, ptNum, pixPosLi):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		posImgLi = []
		for pos in pixPosLi:
			imPos = self.GetPixelPos(ptNum, pos[0], pos[1])
			posImgLi.append(imPos)
		imLoc = np.array(posImgLi, dtype=np.float64)
		
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 1)[0]

	def GetPixelImPos(self, x, y):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		imPos = self.GetPixelPosImPos(x, y)
		imLoc = np.array([imPos], dtype=np.float64)
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 1)[0][0]

	def GetPixelsImPos(self, pixPosLi):

		#Lazy load of image
		if self.imarr is None:
			self.LoadImage()

		posImgLi = []
		for pos in pixPosLi:
			imPos = self.GetPixelPosImPos(pos[0], pos[1])
			posImgLi.append(imPos)
		imLoc = np.array(posImgLi, dtype=np.float64)
		
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 1)[0]

	def NumPoints(self):
		return len(self.model)

class KernelFilter:
	def __init__(self, normImIn):
		self.kernel = [[1,0,-1],[2,0,-2],[1,0,-1]]
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

	def GetPixelImPos(self, xOff, yOff):
		total = 0.
		for x in range(-self.halfw, self.halfw+1):
			for y in range(-self.halfw, self.halfw+1):
				comp = self.kernel[y+self.halfw][x+self.halfw]
				total += self.normIm.GetPixelImPos(self.scale*x+xOff, self.scale*y+yOff) * comp
		#print xOff, yOff, total
		if self.absVal:
			return np.abs(total)
		return total

	def GetPixelsImPos(self, pixPosLi):
		out = []
		for pos in pixPosLi:
			out.append(self.GetPixelImPos(pos[0], pos[1]))
		return out

def ExtractPatch(normImage, ptNum, xOff, yOff, patchw=24, patchh=24, scale=0.08):

	localPatch = np.zeros((patchh, patchw, 3), dtype=np.uint8)
	for x in range(patchw):
		for y in range(patchh):
			localPatch[y,x,:] = normImage.GetPixel(ptNum, \
				(x-((patchw-1)/2))*scale+xOff, \
				(y-((patchh-1)/2))*scale+yOff)
	return localPatch

def ExtractPatchAtImg(normImage, ptX, ptY, patchw=24, patchh=24, scale=0.08):

	localPatch = np.zeros((patchh, patchw, 3), dtype=np.uint8)
	for x in range(patchw):
		for y in range(patchh):
			localPatch[y,x,:] = normImage.GetPixelImPos(
				(x-((patchw-1)/2))*scale+ptX, \
				(y-((patchh-1)/2))*scale+ptY)
	return localPatch


