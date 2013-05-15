
import urllib2
import json, pickle, math, StringIO, random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import procrustes, pxutil
import normalisedImage
from sklearn.ensemble import GradientBoostingRegressor

def ExtractSupportIntensity(normImage, supportPixOff, ptNum, offX, offY):
	supportPixOff = supportPixOff.copy()
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
		
	trainNormSamples = normalisedSamples[:400]
	testNormSamples = normalisedSamples[400:]

	#print normalisedSamples[0].model
	#print normalisedSamples[0].GetPixelPos(0, 0, 0)
	#print normalisedSamples[0].params
	#print normalisedSamples[0].GetPixel(0, 0., 0)

	supportPixOff = np.random.uniform(low=-0.7, high=0.7, size=(50, 2))
	
	trainInt = []
	trainOff = []

	while len(trainOff) < 1000:
		x = np.random.normal(scale=0.5)
		print len(trainOff), x
		
		pix, valid = ExtractSupportIntensity(random.sample(trainNormSamples,1)[0], supportPixOff, 0, 0.+x, 0.)
		if sum(valid) != len(valid): continue
		pixGrey = [pxutil.ToGrey(p) for p in pix]

		#print pixGrey
		trainInt.append(pixGrey)
		trainOff.append(x)

	reg = GradientBoostingRegressor()
	reg.fit(trainInt, trainOff)

	#trainPred = reg.predict(trainInt)
	#plt.plot(trainOff, trainPred, 'x')
	#plt.show()

	testOff = []
	testPred = []
	while len(testOff) < 100:
		x = np.random.normal(scale=0.5)
		print len(testOff), x

		pix, valid = ExtractSupportIntensity(random.sample(testNormSamples,1)[0], supportPixOff, 0, 0.+x, 0.)
		if sum(valid) != len(valid): continue
		pixGrey = [pxutil.ToGrey(p) for p in pix]

		pred = reg.predict([pixGrey])
		#print x, pred, valid, sum(valid)
		testOff.append(x)
		testPred.append(pred)

	print np.corrcoef(testOff, testPred)[0,1]
	plt.plot(testOff, testPred, 'x')
	plt.show()



