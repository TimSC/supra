
import urllib2
import StringIO, procrustes, pxutil
from PIL import Image
import numpy as np

class NormalisedImage:
	def __init__(self, urlIn, modelIn, meanFaceIn):
		self.url = urlIn
		self.model = modelIn
		self.meanFace = meanFaceIn
		self.im, self.iml, self.imarr = None, None, None
		self.procShape = None
		self.params = None
		
	def GetPixelPos(self, ptNum, x, y):

		#Lazy procrustes calculation
		modelArr = np.array(self.model)
		if self.params is None:
			self.procShape, self.params = procrustes.CalcProcrustesOnFrame(\
				procrustes.FrameToArray(modelArr),\
				procrustes.FrameToArray(self.meanFace))
		
		#Translate in normalised space, then convert back to image space
		startPoint = self.procShape[ptNum]
		modPoint = [startPoint[0]+x, startPoint[1]+y]
		imgPos = procrustes.ToImageSpace(np.array([modPoint]), self.params)
		return imgPos[0]

	def GetPixel(self, ptNum, x, y):

		#Lazy load of image
		if self.im is None:
			urlImgHandle = urllib2.urlopen(self.url)
			self.im = Image.open(StringIO.StringIO(urlImgHandle.read()))
			#self.iml = self.im.load()
			self.imarr = np.array(self.im)

		imPos = self.GetPixelPos(ptNum, x, y)
		imLoc = np.array([imPos], dtype=np.float64)
		
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc)[0]

	def GetPixels(self, ptNum, pixPosLi):

		#Lazy load of image
		if self.im is None:
			urlImgHandle = urllib2.urlopen(self.url)
			self.im = Image.open(StringIO.StringIO(urlImgHandle.read()))
			#self.iml = self.im.load()
			self.imarr = np.array(self.im)

		posImgLi = []
		for pos in pixPosLi:
			imPos = self.GetPixelPos(ptNum, pos[0], pos[1])
			posImgLi.append(imPos)
		imLoc = np.array(posImgLi, dtype=np.float64)
		
		return pxutil.GetPixIntensityAtLoc(self.imarr, imLoc)


