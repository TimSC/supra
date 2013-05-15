
import urllib2
import StringIO, procrustes
from PIL import Image
import numpy as np

class NormalisedImage:
	def __init__(self, urlIn, modelIn, meanFaceIn):
		self.url = urlIn
		self.model = modelIn
		self.meanFace = meanFaceIn
		self.im, self.iml = None, None
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
			self.iml = self.im.load()

		imPos = self.GetPixelPos(ptNum, x, y)
		return self.iml[imPos[0], imPos[1]]

