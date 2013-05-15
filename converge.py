
import urllib2
import json, pickle, math, StringIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import procrustes
import normalisedImage


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
		
	print normalisedSamples[0].model
	print normalisedSamples[0].GetPixelPos(0, 0.1, 0)
	print normalisedSamples[0].params

