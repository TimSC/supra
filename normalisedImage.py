
import urllib2
import StringIO, procrustes, procrustesopt, pxutil, math
from PIL import Image
import numpy as np

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

		posImgLi = []
		for pos in pixPosLi:
			imPos = self.GetPixelPosImPos(pos[0], pos[1])
			posImgLi.append(imPos)
		imLoc = np.array(posImgLi, dtype=np.float64)
		
		pix = np.empty((imLoc.shape[0], self.imarr.shape[2]))
		valid = np.empty(imLoc.shape[0], dtype=np.int)
		temp = np.empty((4, self.imarr.shape[2]))
		pxutil.GetPixIntensityAtLoc(self.imarr, imLoc, 2, temp, pix, valid)
		return pix

	def NumPoints(self):
		return min(len(self.model), len(self.procShape))

	def GetParams(self):
		return self.params

	def NumChannels(self):
		return self.imarr.shape[2]

def ExtractPatch(normImage, ptNum, xOff, yOff, patchw=24, patchh=24, scale=0.08):

	localPatch = np.zeros((patchh, patchw, 3), dtype=np.uint8)
	for x in range(patchw):
		for y in range(patchh):
			localPatch[y,x,:] = normImage.GetPixel(ptNum, \
				(x-((patchw-1)/2))*scale+xOff, \
				(y-((patchh-1)/2))*scale+yOff)
	return localPatch

def SaveNormalisedImageToFile(sample, fina):
	im = Image.new("RGB",(300,300))
	iml = im.load()
	pos, posIm = [], []
	for x in range(300):
		for y in range(300):
			nx = (x - 150) / 50.
			ny = (y - 150) / 50.
			pos.append((nx,ny))
			posIm.append((x,y))
		
	posInts = sample.GetPixelsImPos(pos)
	for p, px in zip(posIm, posInts):
		#print posIm, px
		iml[p[0], p[1]] = tuple(map(int,map(round,px)))
	im.save(fina)

class HorizontalMirrorNormalisedImage():
	def __init__(self, imgIn, mapping):
		self.img = imgIn
		self.mapping = mapping #Maps from previous index to new index
		
		self.procShape = []
		for i, pt in enumerate(self.img.procShape):
			if i <= len(self.procShape):
				self.procShape.append((-1.,-1.))
			self.procShape[i] = pt

	def GetPixelsImPos(self, pixPosLi):

		pixPosLi = np.copy(pixPosLi) * [-1., 1.]
		out = self.img.GetPixelsImPos(pixPosLi)
		#self.params = self.img.GetParams()
		return out

	def GetPixel(self, ptNum, x, y):
		ptNum2 = self.mapping[ptNum]
		return self.img.GetPixel(ptNum2, -x, y)

	def NumPoints(self):
		return self.img.NumPoints()

	def NumChannels(self):
		return self.img.NumChannels()


