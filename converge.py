
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

def LoadSamplesFromServer():
	urlHandle = urllib2.urlopen("http://192.168.1.2/photodb/getsamples.php")
	sampleJson = urlHandle.read()

	meanFace = pickle.load(open("meanFace.dat", "rb"))
	sampleList = json.loads(sampleJson)
	normalisedSamples = []

	for sample in sampleList:
		if sample['model'] is None: continue
		url = "http://192.168.1.2/photodb/roiimg.php?roiId="+str(sample['roiId'])
		normalisedSamples.append(normalisedImage.NormalisedImage(url, sample['model'], meanFace, sample))
		
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
		im = Image.new("RGB",(300,300))
		iml = im.load()
		pos, posIm = [], []
		for x in range(300):
			for y in range(300):
				nx = (x - 150) / 50.
				ny = (y - 150) / 50.
				pos.append((nx,ny))
				posIm.append((x,y))
		
		posInts, intValid = sample.GetPixelsImPos(pos)
		for p, px in zip(posIm, posInts):
			#print posIm, px
			iml[p[0], p[1]] = tuple(map(int,map(round,px)))
		im.save("img{0}.jpg".format(i))

class PcaNormImageIntensity():
	def __init__(self, samples):
		self.supportPixOff = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))

		#Get intensity data for each training image
		imgsSparseInt = []
		for sample in samples:
			imgsSparseInt.append(self.ExtractFeatures(sample))

		imgsSparseIntArr = np.array(imgsSparseInt)
		self.meanInt = imgsSparseIntArr.mean(axis=0)
		imgsSparseIntCent = imgsSparseIntArr - self.meanInt
	
		self.u, self.s, self.v = np.linalg.svd(imgsSparseIntCent)

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

	def ExtractFeatures(self, sample):
		imgSparseInt = []
		for pt in range(sample.NumPoints()):
			pix, valid = ExtractSupportIntensity(sample, self.supportPixOff, pt, 0., 0.)
			assert sum(valid) == len(valid)
			pixGrey = [pxutil.ToGrey(p) for p in pix]
			imgSparseInt.extend(pixGrey)		
		return imgSparseInt

	def ProjectToPca(self, sample):
		
		feat = self.ExtractFeatures(sample)
		centred = feat - self.meanInt

		return np.dot(centred, self.v.transpose()) / self.s		

class PcaNormShape():
	def __init__(self, samples):

		#Get shape data for each training image
		shapes = []
		for sample in samples:
			sampleModel = np.array(sample.model)
			sampleModel = sampleModel.reshape(sampleModel.size)
			shapes.append(sampleModel)

		shapesArr = np.array(shapes)
		self.meanShape = shapesArr.mean(axis=0)
		meanShapeCent = shapesArr - self.meanShape
	
		self.u, self.s, self.v = np.linalg.svd(meanShapeCent)

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

	def ProjectToPca(self, sample):
		
		sampleModel = np.array(sample.model)
		sampleModel = sampleModel.reshape(sampleModel.size)

		centred = sampleModel - self.meanShape

		return np.dot(centred, self.v.transpose()) / self.s

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

	if 0:
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
	pcaShape = PcaNormShape(filteredSamples)
	pcaInt = PcaNormImageIntensity(filteredSamples)

	#DumpNormalisedImages(filteredSamples)

	trainInt = []
	trainOff = []

	for sample in trainNormSamples:

		eigenPcaInt = pcaInt.ProjectToPca(sample)[:20]
		eigenShape = pcaShape.ProjectToPca(sample)[:5]

		for count in range(50):
			x = np.random.normal(scale=0.3)
			print len(trainOff), x

			pix, valid = ExtractSupportIntensity(sample, supportPixOff, 0, 0.+x, 0.)
			if sum(valid) != len(valid): continue
			pixGrey = [pxutil.ToGrey(p) for p in pix]

			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			#print pixGrey
			feat = np.concatenate([pixGreyNorm, eigenPcaInt, eigenShape])

			trainInt.append(feat)
			trainOff.append(x)

	reg = GradientBoostingRegressor()
	reg.fit(trainInt, trainOff)

	#trainPred = reg.predict(trainInt)
	#plt.plot(trainOff, trainPred, 'x')
	#plt.show()

	testOff = []
	testPred = []
	for sample in testNormSamples:
		eigenPcaInt = pcaInt.ProjectToPca(sample)[:20]
		eigenShape = pcaShape.ProjectToPca(sample)[:5]

		for count in range(3):
			x = np.random.normal(scale=0.3)
			print len(testOff), x

			pix, valid = ExtractSupportIntensity(sample, supportPixOff, 0, 0.+x, 0.)
			if sum(valid) != len(valid): continue
			pixGrey = [pxutil.ToGrey(p) for p in pix]

			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			#print pixGrey
			feat = np.concatenate([pixGreyNorm, eigenPcaInt, eigenShape])

			pred = reg.predict([feat])[0]
			#print x, pred, valid, sum(valid)
			testOff.append(x)
			testPred.append(pred)

	correl = np.corrcoef(np.array([testOff]), np.array([testPred]))[0,1]
	print "correl",correl

	signX = SignAgreement(testOff, testPred)
	print "signScore",signX

	log.write(str(correl)+",\t"+str(signScore)+"\n")
	log.flush()
	#plt.plot(testOff, testPred, 'x')
	#plt.show()

if __name__ == "__main__":

	log = open("shapeandint.txt","wt")
	while 1:
		RunTest(log)
	
