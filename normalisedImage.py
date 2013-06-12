
import urllib2
import StringIO, procrustes, procrustesopt, pxutil, math
from PIL import Image
import numpy as np

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
		self.procShape = None
		self.GetProcrustesNormedModel()

	def GetPixelImPos(self, x, y, out = None):
		return self.img.GetPixelImPos(-x, y, out)

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

	def GetProcrustesNormedModel(self):
		if self.procShape is None:
			self.procShape = []
			originalShape = self.img.GetProcrustesNormedModel()
			for i, pt in enumerate(originalShape):
				if i <= len(originalShape):
					self.procShape.append((-1.,-1.))
				self.procShape[i] = pt
		return self.procShape

