
import urllib2
import json, pickle, math, StringIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import procrustes

class NormalisedImage:
	def __init__(self, urlIn, modelIn):
		self.url = url
		self.model = modelIn
		self.im = None
		self.procShape = None
		self.params = None
		
	def GetPixelPos(self, ptNum, x, y):
		#Lazy load of image
		if self.im is not None:
			urlImgHandle = urllib2.urlopen(self.url)
			self.im = Image.open(StringIO.StringIO(urlImgHandle.read()))

		#Lazy procrustes calculation
		modelArr = np.array(self.model)
		if self.params is None:
			self.procShape, self.params = procrustes.CalcProcrustesOnFrame(\
				procrustes.FrameToArray(modelArr),\
				procrustes.FrameToArray(meanFace))
		
		#Translate in normalised space, then convert back to image space
		startPoint = self.procShape[ptNum]
		modPoint = [startPoint[0]+x, startPoint[1]+y]
		imgPos = procrustes.ToImageSpace(np.array([modPoint]), self.params)
		return imgPos

if __name__ == "__main__":

	urlHandle = urllib2.urlopen("http://192.168.1.2/photodb/getsamples.php")
	sampleJson = urlHandle.read()

	meanFace = pickle.load(open("meanFace.dat", "rb"))

	sampleList = json.loads(sampleJson)
	normalisedSamples = []

	for sample in sampleList:
		if sample['model'] is None: continue
		url = "http://192.168.1.2/photodb/roiimg.php?roiId="+str(sample['roiId'])
		normalisedSamples.append(NormalisedImage(url, sample['model']))
		
	print normalisedSamples[0].model
	print normalisedSamples[0].GetPixelPos(0, 0.1, 0)
	print normalisedSamples[0].params

