
import numpy as np, pickle, random, pxutil, copy, math, normalisedImage, converge
import skimage.color as col, skimage.feature as feature, skimage.filter as filt
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def ExtractSupportIntensity(normImage, supportPixOff, ptX, ptY, offX, offY):
	supportPixOff = supportPixOff.copy()
	supportPixOff += [offX, offY]
	supportPixOff += [ptX, ptY]
	return normImage.GetPixelsImPos(supportPixOff)

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

class SupraAxis():
	def __init__(self, axisXIn = 1., axisYIn = 0.):
		pass
	
	def PrepareModel(self, features, offsets):
		pass

class SupraAxisSet():

	def __init__(self, ptNumIn):
		self.ptNum = ptNumIn
		self.supportPixOff = None
		self.supportPixOffSobel = None
		self.trainInt = []
		self.trainOffX, self.trainOffY = [], []
		self.regX, self.regY = None, None

	def AddTraining(self, sample, trainOffset):

		if self.supportPixOff is None:
			self.supportPixOff = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))
		if self.supportPixOffSobel is None:
			self.supportPixOffSobel = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))

		xOff = trainOffset[self.ptNum][0]
		yOff = trainOffset[self.ptNum][1]
		sobelSample = normalisedImage.KernelFilter(sample)

		ptX, ptY = sample.procShape[self.ptNum][0], sample.procShape[self.ptNum][1]
		pix = ExtractSupportIntensity(sample, self.supportPixOff, ptX, ptY, xOff, yOff)
		pixGrey = np.array([ColConv(p) for p in pix])
		pixGrey = pixGrey.reshape(pixGrey.size)

		pixGreyNorm = np.array(pixGrey)
		pixGreyNorm -= pixGreyNorm.mean()

		pixSobel = ExtractSupportIntensity(sobelSample, self.supportPixOffSobel, ptX, ptY, xOff, yOff)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)

		pixNormSobel = np.array(pixConvSobel)
		pixNormSobel -= pixNormSobel.mean()

		localPatch = col.rgb2grey(normalisedImage.ExtractPatch(sample, self.ptNum, xOff, yOff))
		hog = feature.hog(localPatch)

		#print pixGrey
		feat = np.concatenate([pixGreyNorm, hog, self.holisiticFeatures, pixNormSobel])

		self.trainInt.append(feat)
		self.trainOffX.append(xOff)
		self.trainOffY.append(yOff)

	def AddHolisticFeatures(self, feat):
		self.holisiticFeatures = feat

	def PrepareModel(self):
		self.regX = GradientBoostingRegressor()
		self.regX.fit(self.trainInt, self.trainOffX)
		self.regY = GradientBoostingRegressor()
		self.regY.fit(self.trainInt, self.trainOffY)

	def Predict(self, sample, model, prevFrameFeatures):
		sobelSample = normalisedImage.KernelFilter(sample)

		pix = ExtractSupportIntensity(sample, self.supportPixOff, model[self.ptNum][0], model[self.ptNum][1], 0., 0.)
		pixGrey = np.array([ColConv(p) for p in pix])
		pixGrey = pixGrey.reshape(pixGrey.size)
			
		pixGreyNorm = np.array(pixGrey)
		pixGreyNorm -= pixGreyNorm.mean()

		pixSobel = ExtractSupportIntensity(sobelSample, self.supportPixOffSobel, model[self.ptNum][0], model[self.ptNum][1], 0., 0.)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)

		pixNormSobel = np.array(pixConvSobel)
		pixNormSobel -= pixNormSobel.mean()

		localPatch = col.rgb2grey(normalisedImage.ExtractPatchAtImg(sample, model[self.ptNum][0], model[self.ptNum][1]))
		hog = feature.hog(localPatch)

		#print pixGrey
		feat = np.concatenate([pixGreyNorm, hog, prevFrameFeatures, pixNormSobel])

		predX = self.regX.predict([feat])[0]
		predY = self.regY.predict([feat])[0]
		return predX, predY

class SupraCloud():

	def __init__(self, trainNormSamples):
		self.trackers = None
		self.pcaShape = converge.PcaNormShape(trainNormSamples)
		self.pcaInt = converge.PcaNormImageIntensity(trainNormSamples)

	def AddTraining(self, sample, numExamples):
		if self.trackers is None:
			self.trackers = [SupraAxisSet(x) for x in range(sample.NumPoints())]

		eigenPcaInt = self.pcaInt.ProjectToPca(sample, sample.procShape)[:20]
		eigenShape = self.pcaShape.ProjectToPca(sample, sample.procShape)[:5]

		for sampleCount in range(numExamples):
			perturb = []
			for num in range(sample.NumPoints()):
				perturb.append((np.random.normal(scale=0.3),np.random.normal(scale=0.3)))

			for count, tracker in enumerate(self.trackers):
				tracker.AddHolisticFeatures(np.concatenate([eigenPcaInt, eigenShape]))
				tracker.AddTraining(sample, perturb)

	def PrepareModel(self):
		for tracker in self.trackers:
			tracker.PrepareModel()

	def ExtractFeatures(self, sample, model):
		pass

	def CalcPrevFrameFeatures(self, sample, model):
		eigenPcaInt = self.pcaInt.ProjectToPca(sample, model)[:20]
		eigenShape = self.pcaShape.ProjectToPca(sample, model)[:5]
		return np.concatenate([eigenPcaInt, eigenShape])

	def Predict(self, sample, model, prevFrameFeatures):
		out = []
		for tracker in self.trackers:
			out.append(tracker.Predict(sample, model, prevFrameFeatures))
		return out


def TrainTracker(trainNormSamples):
	cloudTracker = SupraCloud(trainNormSamples)

	for sampleCount, sample in enumerate(trainNormSamples):
		print sampleCount, len(trainNormSamples)
		cloudTracker.AddTraining(sample, 50)

	cloudTracker.PrepareModel()

	return cloudTracker

def TestTracker(cloudTracker, testNormSamples):
	testOffX, testOffY = [], []
	testPredX, testPredY = [], []
	for sample in testNormSamples:

		prevFrameFeat = cloudTracker.CalcPrevFrameFeatures(sample, sample.procShape)

		sobelSample = normalisedImage.KernelFilter(sample)

		for count in range(3):
			x = np.random.normal(scale=0.3)
			y = np.random.normal(scale=0.3)
			
			print len(testOffX), x, y
			testPos = []
			for pt in sample.procShape:
				testPos.append((pt[0] + x, pt[1] + y))

			predX, predY = cloudTracker.Predict(sample, testPos, prevFrameFeat)[0]

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
	return cloudTracker

if __name__ == "__main__":

	if 0:
		normalisedSamples = LoadSamplesFromServer()
		pickle.dump(normalisedSamples, open("normalisedSamples.dat","wb"), protocol=-1)
	else:
		normalisedSamples = pickle.load(open("normalisedSamples.dat","rb"))		

	#Only use larger faces
	filteredSamples = []
	for sample in normalisedSamples:
		if np.array(sample.model).std(axis=1)[0] > 15.:
			filteredSamples.append(sample)

	#DumpNormalisedImages(filteredSamples)

	#Reduce problem to one point
	for sample in filteredSamples:
		sample.procShape = sample.procShape[0:1,:]

	log = open("log.txt","wt")

	while 1:
		print "Filtered to",len(filteredSamples),"of",len(normalisedSamples),"samples"
		halfInd = len(filteredSamples)/2
		random.shuffle(filteredSamples)
		trainNormSamples = filteredSamples[:halfInd]
		testNormSamples = filteredSamples[halfInd:]

		if 0:
			cloudTracker = TrainTracker(trainNormSamples)
			pickle.dump(cloudTracker, open("tracker.dat","wb"), protocol=-1)
			pickle.dump(testNormSamples, open("testNormSamples.dat","wb"), protocol=-1)
		else:
			cloudTracker = pickle.load(open("tracker.dat","rb"))
			testNormSamples = pickle.load(open("testNormSamples.dat","rb"))

		#Run performance test
		TestTracker(cloudTracker, testNormSamples)

