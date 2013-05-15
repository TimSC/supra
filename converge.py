
import urllib2
import json, pickle, math, StringIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import procrustes, pxutil
import normalisedImage

def ExtractSupportIntensity(normImage, supportPixOff, ptNum, offX, offY):
	supportPixOff += [offX, offY]
	return normImage.GetPixels(ptNum, supportPixOff)

if __name__ == "__main__":

	urlHandle = urllib2.urlopen("http://192.168.1.2/photodb/getsamples.php")
	sampleJson = urlHandle.read()

	meanFace = pickle.load(open("meanFace.dat", "rb"))

	sampleList = json.loads(sampleJson)
	normalisedSamples = []

	for sample in sampleList:
		if sample['model'] is None: continue
		url = "http://192.168.1.2/photodb/roiimg.php?roiId="+str(sample['roiId'])
		normalisedSamples.append(normalisedImage.NormalisedImage(url, sample['model'], meanFace))
		
	#print normalisedSamples[0].model
	#print normalisedSamples[0].GetPixelPos(0, 0, 0)
	#print normalisedSamples[0].params
	#print normalisedSamples[0].GetPixel(0, 0., 0)

	supportPixOff = np.random.uniform(low=-0.7, high=0.7, size=(50, 2))
	
	pix = ExtractSupportIntensity(normalisedSamples[0], supportPixOff, 0, 0., 0.)
	pixGrey = [pxutil.ToGrey(p) for p in pix]
	print pixGrey

