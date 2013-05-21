
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
		pass

	def AddTraining(self, sample, trainOffset):
		pass

	def AddHolisticFeatures(self, feat):
		pass

	def PrepareModel(self):
		pass

	def Predict(self, sample, model, prevFrameFeatures):
		pass
class SupraCloud():

	def __init__(self):
		pass

	def AddTraining(self, sample, numExamples):
		pass

	def PrepareModel(self):
		pass

	def ExtractFeatures(self, sample, model):
		pass

	def CalcPrevFrameFeatures(self, sample, model):
		pass

	def Predict(self, sample, model, prevFrameFeatures):
		pass


def TrainTracker(trainNormSamples, testNormSamples, log):
	cloudTracker = SupraCloud()
	
	supportPixOff = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))
	supportPixOffSobel = np.random.uniform(low=-0.3, high=0.3, size=(50, 2))
	pcaShape = converge.PcaNormShape(filteredSamples)
	pcaInt = converge.PcaNormImageIntensity(filteredSamples)

	#DumpNormalisedImages(filteredSamples)

	trainInt = []
	trainOffX, trainOffY = [], []

	for sample in trainNormSamples:

		eigenPcaInt = pcaInt.ProjectToPca(sample)[:20]
		eigenShape = pcaShape.ProjectToPca(sample)[:5]
		sobelSample = normalisedImage.KernelFilter(sample)

		for count in range(50):
			x = np.random.normal(scale=0.3)
			y = np.random.normal(scale=0.3)
			print len(trainOffX), x, y

			ptX, ptY = sample.procShape[0][0], sample.procShape[0][1]
			pix = ExtractSupportIntensity(sample, supportPixOff, ptX, ptY, x, y)
			pixGrey = np.array([ColConv(p) for p in pix])
			pixGrey = pixGrey.reshape(pixGrey.size)

			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			pixSobel = ExtractSupportIntensity(sobelSample, supportPixOffSobel, ptX, ptY, x, y)
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
		sobelSample = normalisedImage.KernelFilter(sample)

		for count in range(3):
			x = np.random.normal(scale=0.3)
			y = np.random.normal(scale=0.3)
			print len(testOffX), x, y

			ptX, ptY = sample.procShape[0][0], sample.procShape[0][1]
			pix = ExtractSupportIntensity(sample, supportPixOff, ptX, ptY, x, y)
			pixGrey = np.array([ColConv(p) for p in pix])
			pixGrey = pixGrey.reshape(pixGrey.size)
			
			pixGreyNorm = np.array(pixGrey)
			pixGreyNorm -= pixGreyNorm.mean()

			pixSobel = ExtractSupportIntensity(sobelSample, supportPixOffSobel, ptX, ptY, x, y)
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

		if 1:
			cloudTracker = TrainTracker(trainNormSamples, testNormSamples, log)
			pickle.dump(cloudTracker, open("tracker.dat","wb"), protocol=-1)
			pickle.dump(testNormSamples, open("testNormSamples.dat","wb"), protocol=-1)
		else:
			cloudTracker = pickle.load(open("tracker.dat","rb"))
			testNormSamples = pickle.load(open("testNormSamples.dat","rb"))

		#Run performance test

		testVal, testErr = [], []
		for sampleNum, sample in enumerate(testNormSamples):
			print sampleNum
		
			for count in range(1):

				prevFrameFeat = cloudTracker.CalcPrevFrameFeatures(sample, sample.procShape)
				#print sample.procShape

				testOffset = []
				modProcShape = copy.deepcopy(sample.procShape)
				for pt in range(sample.procShape.shape[0]):
					x = np.random.normal(scale=0.3)
					y = 0. #np.random.normal(scale=0.3) #SIMP
					testOffset.append((x,y))
					modProcShape[pt,0] += x
					modProcShape[pt,1] += y
			
				#print modProcShape
			
				pred = cloudTracker.Predict(sample, modProcShape, prevFrameFeat)
			
				for testOff, actualPt, predPt in zip(testOffset, sample.procShape, pred):
					error = actualPt - predPt
					#print testOff, actualPt, predPt, error
					testVal.append(testOff[0])
					testErr.append(pred[0])
					#testVal.append(testOff[1])
					#testErr.append(error[1])
	
		correl = np.corrcoef([testVal, testErr])[0,1]
		print "correl", correl
		log.write(str(correl)+"\n")
		log.flush()
		#plt.plot(testVal, testErr, 'bx')
		#plt.show()


