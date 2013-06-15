
import urllib2
import json, pickle, math, StringIO, random
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import procrustes, pxutil
import normalisedImage, normalisedImageOpt
from sklearn.ensemble import GradientBoostingRegressor
import skimage.color as col, skimage.feature as feature, skimage.filter as filt

def ExtractSupportIntensity(normImage, supportPixOff, ptNum, offX, offY):
	supportPixOff = supportPixOff.copy()
	supportPixOff += [offX, offY]
	return normImage.GetPixels(ptNum, supportPixOff)

def ExtractSupportIntensityAsImg(normImage, supportPixOff, ptX, ptY):
	supportPixOff = supportPixOff.copy()
	supportPixOff += [ptX, ptY]
	return normImage.GetPixelsImPos(supportPixOff)

def LoadSamplesFromServer():
	urlHandle = urllib2.urlopen("http://192.168.1.9/photodb/getsamples.php")
	sampleJson = urlHandle.read()

	meanFace = pickle.load(open("meanFace.dat", "rb"))
	sampleList = json.loads(sampleJson)
	normalisedSamples = []

	for sample in sampleList:
		if sample['model'] is None: continue
		url = "http://192.168.1.9/photodb/roiimg.php?roiId="+str(sample['roiId'])
		normalisedSamples.append(normalisedImageOpt.NormalisedImage(url, sample['model'], meanFace, sample))
		
	trainNormSamples = normalisedSamples[:400]
	testNormSamples = normalisedSamples[400:]

	for i, img in enumerate(normalisedSamples):
		print i, len(normalisedSamples)
		img.LoadImage()
		img.CalcProcrustes()
		img.ClearPilImage()
	return normalisedSamples

def DumpNormalisedImages(filteredSamples):
	for i, sample in enumerate(filteredSamples):
		print i
		normalisedImage.SaveNormalisedImageToFile(sample, "img{0}.jpg".format(i))

class PcaNormImageIntensity():
	def __init__(self, samples):
		self.supportPixOff = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))

		#Get intensity data for each training image
		imgsSparseInt = []
		for sample in samples:
			imgsSparseInt.append(self.ExtractFeatures(sample, sample.procShape))

		imgsSparseIntArr = np.array(imgsSparseInt)
		self.meanInt = imgsSparseIntArr.mean(axis=0)
		imgsSparseIntCent = imgsSparseIntArr - self.meanInt
	
		self.u, self.s, self.v = np.linalg.svd(imgsSparseIntCent, full_matrices=False)

		#print imgsSparseIntCent

		#print self.u.shape
		#print self.s.shape
		#print self.v.shape

		self.S = np.zeros((self.u.shape[0], self.v.shape[0]))
		self.S[:self.v.shape[0], :self.v.shape[0]] = np.diag(self.s)

		#reconstructed = np.dot(self.u, np.dot(self.S, self.v))

		#print self.u[0,:]
		#plt.plot(self.s)
		#plt.show()

	def ExtractFeatures(self, sample, model):
		imgSparseInt = []
		for pt in model:
			pix = ExtractSupportIntensityAsImg(sample, self.supportPixOff, pt[0], pt[1])
			pixGrey = [pxutil.ToGrey(p) for p in pix]
			imgSparseInt.extend(pixGrey)		
		return imgSparseInt

	def ProjectToPca(self, sample, model):
		feat = self.ExtractFeatures(sample, model)
		centred = feat - self.meanInt
		ret = np.dot(centred, self.v.transpose()) / self.s
		return ret

class PcaNormShape():
	def __init__(self, samples):

		#Get shape data for each training image
		shapes = []
		for sample in samples:
			sampleModel = np.array(sample.procShape)
			sampleModel = sampleModel.reshape(sampleModel.size)
			shapes.append(sampleModel)

		shapesArr = np.array(shapes)
		self.meanShape = shapesArr.mean(axis=0)
		meanShapeCent = shapesArr - self.meanShape
	
		self.u, self.s, self.v = np.linalg.svd(meanShapeCent, full_matrices=False)

		#print meanShapeCent

		#print self.u.shape
		#print self.s.shape
		#print self.v.shape

		self.S = np.zeros((self.u.shape[0], self.v.shape[0]))
		self.S[:self.v.shape[0], :self.v.shape[0]] = np.diag(self.s)

		#reconstructed = np.dot(self.u, np.dot(self.S, self.v))
		#print reconstructed

		#print self.u[0,:]
		#plt.plot(self.s)
		#plt.show()

	def ProjectToPca(self, sample, model):
		
		sampleModel = np.array(model)
		sampleModel = sampleModel.reshape(sampleModel.size)

		centred = sampleModel - self.meanShape

		return np.dot(centred, self.v.transpose()) / self.s

def ColConv(px):
	out = col.rgb2xyz([[px]])[0][0]
	return out

def SignAgreement(testOff, testPred):
	signTotal = 0
	for tru, pred in zip(testOff, testPred):
		truSign = tru >= 0.
		predSign = pred >= 0.
		if truSign == predSign:
			signTotal += 1
	signScore = float(signTotal) / len(testOff)
	return signScore

def RunTest(log):

	if 1:
		normalisedSamples = LoadSamplesFromServer()
		pickle.dump(normalisedSamples, open("normalisedSamples.dat","wb"), protocol=-1)
	else:
		normalisedSamples = pickle.load(open("normalisedSamples.dat","rb"))		

	#print normalisedSamples[0].model
	#print normalisedSamples[0].GetPixelPos(0, 0, 0)
	#print normalisedSamples[0].params
	#print normalisedSamples[0].GetPixel(0, 0., 0)

	#Only use larger faces
	filteredSamples = []
	for sample in normalisedSamples:
		if np.array(sample.model).std(axis=1)[0] > 15.:
			filteredSamples.append(sample)

	print "Filtered to",len(filteredSamples),"of",len(normalisedSamples),"samples"
	halfInd = len(filteredSamples)/2
	random.shuffle(filteredSamples)
	trainNormSamples = filteredSamples[:halfInd]
	testNormSamples = filteredSamples[halfInd:]

	supportPixOff = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))
	supportPixOffSobel = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))
	pcaShape = PcaNormShape(filteredSamples)
	pcaInt = PcaNormImageIntensity(filteredSamples)

	#DumpNormalisedImages(filteredSamples)

	trainInt = []
	trainOffX, trainOffY = [], []

	for sample in trainNormSamples:

		eigenPcaInt = pcaInt.ProjectToPca(sample, sample.GetProcrustesNormedModel())[:20]
		eigenShape = pcaShape.ProjectToPca(sample, sample.GetProcrustesNormedModel())[:5]
		sobelSample = normalisedImageOpt.KernelFilter(sample)

		for count in range(50):
			x = np.random.normal(scale=0.3)
			y = np.random.normal(scale=0.3)
			print len(trainOffX), x, y

			pix = ExtractSupportIntensity(sample, supportPixOff, 0, x, y)
			pixGrey = np.array([ColConv(p) for p in pix])
			pixGrey = pixGrey.reshape(pixGrey.size)

			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			pixSobel = ExtractSupportIntensity(sobelSample, supportPixOffSobel, 0, x, y)
			pixConvSobel = []
			for px in pixSobel:
				pixConvSobel.extend(px)

			pixNormSobel = np.array(pixConvSobel)
			pixNormSobel -= pixNormSobel.mean()

			localPatch = col.rgb2grey(normalisedImage.ExtractPatch(sample, 0, x, y))
			hog = feature.hog(localPatch)

			#print pixGrey
			feat = np.concatenate([pixGreyNorm, eigenPcaInt, eigenShape, hog, pixNormSobel])

			trainInt.append(feat)
			trainOffX.append(x)
			trainOffY.append(y)

	regX = GradientBoostingRegressor()
	regX.fit(trainInt, trainOffX)
	regY = GradientBoostingRegressor()
	regY.fit(trainInt, trainOffY)

	#trainPred = reg.predict(trainInt)
	#plt.plot(trainOff, trainPred, 'x')
	#plt.show()

	testOffX, testOffY = [], []
	testPredX, testPredY = [], []
	for sample in testNormSamples:
		eigenPcaInt = pcaInt.ProjectToPca(sample)[:20]
		eigenShape = pcaShape.ProjectToPca(sample)[:5]
		sobelSample = normalisedImageOpt.KernelFilter(sample)

		for count in range(3):
			x = np.random.normal(scale=0.3)
			y = np.random.normal(scale=0.3)
			print len(testOffX), x, y

			pix = ExtractSupportIntensity(sample, supportPixOff, 0, x, y)
			pixGrey = np.array([ColConv(p) for p in pix])
			pixGrey = pixGrey.reshape(pixGrey.size)
			
			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			pixSobel = ExtractSupportIntensity(sobelSample, supportPixOffSobel, 0, x, y)
			pixConvSobel = []
			for px in pixSobel:
				pixConvSobel.extend(px)

			pixNormSobel = np.array(pixConvSobel)
			pixNormSobel -= pixNormSobel.mean()

			localPatch = col.rgb2grey(normalisedImage.ExtractPatch(sample, 0, x, y))
			hog = feature.hog(localPatch)

			#print pixGrey
			feat = np.concatenate([pixGreyNorm, eigenPcaInt, eigenShape, hog, pixNormSobel])

			predX = regX.predict([feat])[0]
			predY = regY.predict([feat])[0]
			#print x, pred
			testOffX.append(x)
			testOffY.append(y)
			testPredX.append(predX)
			testPredY.append(predY)

	correlX = np.corrcoef(np.array([testOffX]), np.array([testPredX]))[0,1]
	correlY = np.corrcoef(np.array([testOffY]), np.array([testPredY]))[0,1]
	correl = 0.5*(correlX+correlY)
	print "correl",correl

	signX = SignAgreement(testOffX, testPredX)
	signY = SignAgreement(testOffY, testPredY)
	signScore = 0.5 * (signX + signY)
	print "signScore",signX

	log.write(str(correl)+",\t"+str(signScore)+"\n")
	log.flush()
	#plt.plot(testOff, testPred, 'x')
	#plt.show()

if __name__ == "__main__":

	log = open("all.txt","wt")
	while 1:
		RunTest(log)
	
