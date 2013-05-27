
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
		self.reg = GradientBoostingRegressor()
		self.x = axisXIn
		self.y = axisYIn
	
	def PrepareModel(self, features, offsets):
		offsets = np.array(offsets)
		labels = offsets[:,0] * self.x + offsets[:,1] * self.y
		self.reg.fit(features, labels)
	
class SupraAxisSet():

	def __init__(self, ptNumIn, supportPixHalfWidthIn = 0.3):
		self.ptNum = ptNumIn
		self.supportPixOff = None
		self.supportPixOffSobel = None
		self.trainInt = []
		self.trainOffX, self.trainOffY = [], []
		self.regX, self.regY = None, None
		self.supportPixHalfWidth = supportPixHalfWidthIn
		self.numSupportPix = 50

	def AddTraining(self, sample, trainOffset, extraFeatures):

		if self.supportPixOff is None:
			self.supportPixOff = np.random.uniform(low=-self.supportPixHalfWidth, \
				high=self.supportPixHalfWidth, size=(self.numSupportPix, 2))
		if self.supportPixOffSobel is None:
			self.supportPixOffSobel = np.random.uniform(low=-self.supportPixHalfWidth, \
				high=self.supportPixHalfWidth, size=(self.numSupportPix, 2))

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

		feat = np.concatenate([pixGreyNorm, hog, extraFeatures, pixNormSobel])

		self.trainInt.append(feat)
		self.trainOffX.append(xOff)
		self.trainOffY.append(yOff)

	def PrepareModel(self):
		self.axes = []
		self.axes.append(SupraAxis(1., 0.))
		self.axes.append(SupraAxis(0., 1.))
		trainOff = np.vstack([self.trainOffX, self.trainOffY]).transpose()
		
		for axis in self.axes:
			axis.PrepareModel(self.trainInt, trainOff)

	def Predict(self, sample, model, prevFrameFeatures):
		sobelSample = normalisedImage.KernelFilter(sample)

		pix = ExtractSupportIntensity(sample, self.supportPixOff, \
			model[self.ptNum][0], model[self.ptNum][1], 0., 0.)
		pixGrey = np.array([ColConv(p) for p in pix])
		pixGrey = pixGrey.reshape(pixGrey.size)
			
		pixGreyNorm = np.array(pixGrey)
		pixGreyNorm -= pixGreyNorm.mean()

		pixSobel = ExtractSupportIntensity(sobelSample, self.supportPixOffSobel, \
			model[self.ptNum][0], model[self.ptNum][1], 0., 0.)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)

		pixNormSobel = np.array(pixConvSobel)
		pixNormSobel -= pixNormSobel.mean()

		localPatch = col.rgb2grey(normalisedImage.ExtractPatchAtImg(sample, \
			model[self.ptNum][0], model[self.ptNum][1]))
		hog = feature.hog(localPatch)

		feat = np.concatenate([pixGreyNorm, hog, prevFrameFeatures, pixNormSobel])

		totalx, totaly, weightx, weighty = 0., 0., 0., 0.
		for axis in self.axes:
			pred = axis.reg.predict([feat])[0]
			totalx += pred * axis.x
			totaly += pred * axis.y
			weightx += axis.x
			weighty += axis.y

		return totalx / weightx, totaly / weighty

class SupraCloud():

	def __init__(self, supportPixHalfWidthIn = 0.3, trainingOffsetIn = 0.3):
		self.trackers = None
		self.trainingOffset = trainingOffsetIn #Standard deviations
		self.supportPixHalfWidth = supportPixHalfWidthIn
		self.numIter = 2

	def AddTraining(self, sample, numExamples, extraFeatures):
		if self.trackers is None:
			self.trackers = [SupraAxisSet(x, self.supportPixHalfWidth) \
				for x in range(sample.NumPoints())]

		for sampleCount in range(numExamples):
			perturb = []
			for num in range(sample.NumPoints()):
				perturb.append((np.random.normal(scale=self.trainingOffset),\
					np.random.normal(scale=self.trainingOffset)))

			for count, tracker in enumerate(self.trackers):
				tracker.AddTraining(sample, perturb, extraFeatures)

	def PrepareModel(self):
		for tracker in self.trackers:
			tracker.PrepareModel()

	def Predict(self, sample, model, prevFrameFeatures):

		currentModel = np.array(copy.deepcopy(model))
		for iterNum in range(self.numIter):
			for ptNum, tracker in enumerate(self.trackers):
				pred = tracker.Predict(sample, currentModel, prevFrameFeatures)
				currentModel[ptNum,:] -= pred
		return currentModel

class SupraLayers:
	def __init__(self, trainNormSamples):
		self.numIntPcaComp = 20
		self.numShapePcaComp = 5
		self.pcaShape = converge.PcaNormShape(trainNormSamples)
		self.pcaInt = converge.PcaNormImageIntensity(trainNormSamples)
		self.layers = [SupraCloud(0.3, 0.3),SupraCloud(0.3,0.2),SupraCloud(0.3,0.1)]

	def AddTraining(self, sample, numExamples):
		eigenPcaInt = self.pcaInt.ProjectToPca(sample, sample.procShape)[:self.numIntPcaComp]
		eigenShape = self.pcaShape.ProjectToPca(sample, sample.procShape)[:self.numShapePcaComp]
		extraFeatures = np.concatenate([eigenPcaInt, eigenShape])

		for layer in self.layers:
			layer.AddTraining(sample, numExamples, extraFeatures)

	def PrepareModel(self):
		for layer in self.layers:
			layer.PrepareModel()

	def CalcPrevFrameFeatures(self, sample, model):
		eigenPcaInt = self.pcaInt.ProjectToPca(sample, model)[:self.numIntPcaComp]
		eigenShape = self.pcaShape.ProjectToPca(sample, model)[:self.numShapePcaComp]
		return np.concatenate([eigenPcaInt, eigenShape])

	def Predict(self, sample, model, prevFrameFeatures):
		currentModel = np.array(copy.deepcopy(model))
		for layerNum, layer in enumerate(self.layers):
			currentModel = layer.Predict(sample, currentModel, prevFrameFeatures)
		return currentModel

def TrainTracker(trainNormSamples):
	cloudTracker = SupraLayers(trainNormSamples)

	for sampleCount, sample in enumerate(trainNormSamples):
		print "train", sampleCount, len(trainNormSamples)
		cloudTracker.AddTraining(sample, 50)

	cloudTracker.PrepareModel()
	return cloudTracker

def TestTracker(cloudTracker, testNormSamples, log):
	testOffs = []
	testPredModels = []
	testModels = []
	trueModels = []
	for sampleCount, sample in enumerate(testNormSamples):
		print "test", sampleCount, len(testNormSamples)
		prevFrameFeat = cloudTracker.CalcPrevFrameFeatures(sample, sample.procShape)
		
		print cloudTracker.layers[0].supportPixHalfWidth
		print cloudTracker.layers[1].supportPixHalfWidth
		print cloudTracker.layers[2].supportPixHalfWidth

		for count in range(3):
			#Purturb positions for testing
			testPos = []
			testOff = []
			for pt in sample.procShape:
				x = np.random.normal(scale=0.3)
				y = np.random.normal(scale=0.3)
				testOff.append((x, y))
				testPos.append((pt[0] + x, pt[1] + y))

			#Make predicton
			predModel = cloudTracker.Predict(sample, testPos, prevFrameFeat)

			#Store result
			testOffs.append(testOff)
			testPredModels.append(predModel)
			testModels.append(testPos)
			trueModels.append(sample.procShape)

	#Calculate performance
	testOffs = np.array(testOffs)
	testPredModels = np.array(testPredModels)
	testModels = np.array(testModels)
	trueModels = np.array(trueModels)

	#Calculate relative movement of tracker
	testPreds = []
	for sampleNum in range(testOffs.shape[0]):
		diff = []
		for ptNum in range(testOffs.shape[1]):
			diff.append((testModels[sampleNum,ptNum,0] - testPredModels[sampleNum,ptNum,0], \
				testModels[sampleNum,ptNum,1] - testPredModels[sampleNum,ptNum,1]))
		testPreds.append(diff)
	testPreds = np.array(testPreds)

	#Calculate performance metrics
	correls, signScores = [], []
	for ptNum in range(testOffs.shape[1]):
		correlX = np.corrcoef(testOffs[:,ptNum,0], testPreds[:,ptNum,0])[0,1]
		correlY = np.corrcoef(testOffs[:,ptNum,1], testPreds[:,ptNum,1])[0,1]
		correl = 0.5*(correlX+correlY)
		correls.append(correl)
		#plt.plot(testOffs[:,ptNum,0], testPreds[:,ptNum,0],'x')
		#plt.plot(testOffs[:,ptNum,1], testPreds[:,ptNum,1],'x')
	plt.show()
	
	for ptNum in range(testOffs.shape[1]):
		signX = SignAgreement(testOffs[:,ptNum,0], testPreds[:,ptNum,0])
		signY = SignAgreement(testOffs[:,ptNum,1], testPreds[:,ptNum,1])
		signScore = 0.5 * (signX + signY)
		signScores.append(signScore)

	#Calculate prediction error	
	predErrors, offsetDist = [], []
	for ptNum in range(testOffs.shape[1]):
		errX = trueModels[:,ptNum,0] - testPredModels[:,ptNum,0]
		errY = trueModels[:,ptNum,1] - testPredModels[:,ptNum,1]
		offset = np.power(np.power(testOffs[:,ptNum,0],2.)+np.power(testOffs[:,ptNum,1],2.),0.5)
		err = np.power(np.power(errX,2.)+np.power(errY,2.),0.5)
		predErrors.append(err)
		offsetDist.append(offset)

	#Get average performance
	avCorrel = np.array(correls).mean()
	avSignScore = np.array(signScores).mean()
	predErrors = np.array(predErrors)
	avPredError = predErrors.mean()
	offsetDist = np.array(offsetDist)

	print "correl",avCorrel
	print "signScore",avSignScore
	print "avPredError",avPredError

	#plt.plot(offsetDist[0,:], predErrors[0,:] ,'x')
	#plt.show()

	log.write(str(avCorrel)+","+str(avSignScore)+","+str(avPredError)+"\n")
	log.flush()

def TestSingleExample(cloudTracker, testNormSamples, log):
	pass

if __name__ == "__main__":
	pass

