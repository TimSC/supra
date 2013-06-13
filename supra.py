
import numpy as np, pickle, random, pxutil, copy, math, converge, sqlitedict, os, tempfile
import normalisedImage, normalisedImageOpt, simpleGbrt
import skimage.color as col, skimage.feature as feature, skimage.filter as filt
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def ExtractSupportIntensity(normImage, supportPixOff, ptX, ptY, offX, offY):
	supportPixOff = supportPixOff.copy()
	supportPixOff += [offX, offY]
	supportPixOff += [ptX, ptY]
	return normImage.GetPixelsImPos(supportPixOff)

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

	def GetFeatureImportance(self):
		return self.reg.feature_importances_

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
		self.sobelKernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)
		self.sobelOffsets, halfWidth = normalisedImageOpt.CalcKernelOffsets(self.sobelKernel)
		self.featureGen = None
		self.featureMultiplex = simpleGbrt.FeatureGenTest()
		self.trainIntDb = None

	def __del__(self):
		del self.trainIntDb
		try:
			if self.trainIntDbFina is not None:
				os.remove(self.trainIntDbFina)
		except:
			pass

	def AddTraining(self, sample, trainOffset, extraFeatures):

		if self.featureGen is None:
			self.featureGen = FeatureGen(self.supportPixHalfWidth, self.numSupportPix)
			
		xOff = trainOffset[self.ptNum][0]
		yOff = trainOffset[self.ptNum][1]

		if self.trainIntDb is None:
			self.trainIntDbFina = tempfile.mkstemp()[1]
			self.trainIntDb = sqlitedict.SqliteDict(self.trainIntDbFina, autocommit=True)

		self.featureGen.SetImage(sample)
		self.featureGen.SetModel(sample.procShape)
		self.featureGen.SetPrevFrameFeatures(extraFeatures)
		self.featureGen.SetModelOffset(trainOffset)
		self.featureGen.SetShapeNoise(0.3)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(xOff, yOff)
		self.featureGen.Gen()

		#self.trainInt.append(features)
		self.trainIntDb[str(len(self.trainOffX))] = self.featureGen[:]
		self.trainOffX.append(xOff)
		self.trainOffY.append(yOff)

	def PrepareModel(self):
		self.axes = []
		self.axes.append(SupraAxis(1., 0.))
		self.axes.append(SupraAxis(0., 1.))
		trainOff = np.vstack([self.trainOffX, self.trainOffY]).transpose()
		
		keys = map(int, self.trainIntDb.keys())
		print "Loading",len(keys),"samples for training"
		keys.sort()
		self.trainInt = np.empty((len(keys), len(self.trainIntDb[0])), dtype=np.float32, order='C')
		for k in keys:
			self.trainInt[k, :] = self.trainIntDb[str(k)]

		for axis in self.axes:
			axis.PrepareModel(self.trainInt, trainOff)

		self.trainInt = None

	def ClearTraining(self):
		self.trainInt = None
		self.trainIntDb = None
		try:
			os.remove(self.trainIntDbFina)
		except:
			pass
		self.trainOffX = None
		self.trainOffY = None

	def GetFeatureImportance(self):
		out = []
		for axis in self.axes:
			out.append(axis.GetFeatureImportance())
		return out

	def Predict(self, sample, model, prevFrameFeatures):

		self.featureGen.SetImage(sample)
		self.featureGen.SetModel(model)
		self.featureGen.SetPrevFrameFeatures(prevFrameFeatures)
		self.featureGen.ClearModelOffset()
		self.featureGen.SetShapeNoise(0.)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(0., 0.)
		self.featureGen.Gen()

		self.featureMultiplex.ClearFeatureSets()
		self.featureMultiplex.AddFeatureSet(self.featureGen[:])

		totalx, totaly, weightx, weighty = 0., 0., 0., 0.
		for axis in self.axes:
			pred = simpleGbrt.PredictGbrt(axis.reg, self.featureMultiplex)
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

	def ClearTraining(self):
		for tracker in self.trackers:
			tracker.ClearTraining()

	def GetFeatureImportance(self):
		out = []
		for tracker in self.trackers:
			out.extend(tracker.GetFeatureImportance())		
		return out

	def Predict(self, sample, model, prevFrameFeatures):

		currentModel = np.array(copy.deepcopy(model))
		for iterNum in range(self.numIter):
			for ptNum, tracker in enumerate(self.trackers):
				pred = tracker.Predict(sample, currentModel, prevFrameFeatures)
				currentModel[ptNum,:] -= pred
		return currentModel

class SupraLayers:
	def __init__(self, trainNormSamples):
		self.featureGenPrevFrame = FeatureGenPrevFrame(trainNormSamples, 20, 5)
		self.layers = [SupraCloud(0.3,0.2),SupraCloud(0.3,0.05)]

	def AddTraining(self, sample, numExamples):

		#Add noise to shape for previous frame features
		prevShapePerturb = copy.deepcopy(sample.procShape)
		for ptNum in range(len(prevShapePerturb)):
			pos = prevShapePerturb[ptNum]
			pos[0] += np.random.normal(scale=0.1)
			pos[1] += np.random.normal(scale=0.1)

		#Extract features from synthetic previous frame
		extraFeatures = self.featureGenPrevFrame.Gen(sample, prevShapePerturb)

		for layer in self.layers:
			layer.AddTraining(sample, numExamples, extraFeatures)

	def PrepareModel(self):
		for layer in self.layers:
			layer.PrepareModel()

	def ClearTraining(self):
		for layer in self.layers:
			layer.ClearTraining()

	def GetFeatureImportance(self):
		out = []
		for layer in self.layers:
			out.extend(layer.GetFeatureImportance())
		return out

	def CalcPrevFrameFeatures(self, sample, model):
		#Extract features from synthetic previous frame
		return self.featureGenPrevFrame.Gen(sample, model)

	def Predict(self, sample, model, prevFrameFeatures):
		currentModel = np.array(copy.deepcopy(model))
		for layerNum, layer in enumerate(self.layers):
			currentModel = layer.Predict(sample, currentModel, prevFrameFeatures)
		return currentModel

class FeatureGen:
	def __init__(self, supportPixHalfWidth, numSupportPix=50):
		self.sample = None
		self.feat = None
		self.supportPixOff = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))
		self.supportPixOffSobel = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))

	def SetImage(self, img):
		self.sample = img

	def SetModel(self, model):
		self.model = model

	def SetPrevFrameFeatures(self, prevFeat):
		self.prevFrameFeatures = prevFeat

	def SetModelOffset(self, modelOffset):
		self.modelOffset = modelOffset

	def ClearModelOffset(self):
		self.modelOffset = np.zeros(np.array(self.modelOffset).shape)

	def SetShapeNoise(self, noise):
		self.shapeNoise = noise

	def SetPointNum(self, ptNum):
		self.ptNum = ptNum

	def SetOffset(self, offX, offY):
		self.xOff = offX
		self.yOff = offY

	def GenIntSupport(self, ptNum, xOff, yOff):
		pix = ExtractSupportIntensity(self.sample, self.supportPixOff, \
			self.model[ptNum][0], self.model[ptNum][1], xOff, yOff)
		pix = pix.reshape((pix.shape[0],1,pix.shape[1]))
		pixGrey = col.rgb2xyz(pix)
		pixGrey = pixGrey.reshape(pixGrey.size)
		
		#pixGreyNorm = np.array(pixGrey)
		#pixGreyNorm -= pixGreyNorm.mean()
		return pixGrey

	def GenSobelSupport(self, ptNum, xOff, yOff):
		sobelSample = normalisedImageOpt.KernelFilter(self.sample)
		pixSobel = ExtractSupportIntensity(sobelSample, self.supportPixOffSobel, \
			self.model[ptNum][0], self.model[ptNum][1], xOff, yOff)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)

		#pixNormSobel = np.array(pixConvSobel)
		#pixNormSobel -= pixNormSobel.mean()
		return pixConvSobel

	def GenHog(self, ptNum, xOff, yOff):
		imLocs = normalisedImageOpt.GenPatchOffsetList(self.model[ptNum][0]+xOff, self.model[ptNum][1]+yOff)
		localPatch = normalisedImageOpt.ExtractPatchAtImg(self.sample, imLocs)
		localPatchGrey = col.rgb2grey(np.array([localPatch]))
		localPatchGrey = localPatchGrey.reshape((24,24)).transpose()
		return feature.hog(localPatchGrey)

	def GenDistancePairs(self, ptNum, xOff, yOff):
		out = []
		modifiedPos = np.array(self.model) + np.array(self.modelOffset)

		for i in range(len(self.modelOffset)):
			if i == ptNum: continue	
			dx = (modifiedPos[i,0] - modifiedPos[ptNum,0])
			dy = (modifiedPos[i,1] - modifiedPos[ptNum,1])
			if self.shapeNoise > 0.:
				dx += np.random.randn() * self.shapeNoise
				dy += np.random.randn() * self.shapeNoise
			out.append(dx)
			out.append(dy)
		return out

	def Gen(self):

		pixGreyNorm = self.GenIntSupport(self.ptNum, self.xOff, self.yOff)
		#print pixGreyNorm.shape
		pixNormSobel = self.GenSobelSupport(self.ptNum, self.xOff, self.yOff)
		#print pixNormSobel.shape
		hog = self.GenHog(self.ptNum, self.xOff, self.yOff)
		#print hog.shape
		relDist = self.GenDistancePairs(self.ptNum, self.xOff, self.yOff)

		self.feat = np.concatenate([pixGreyNorm, hog, self.prevFrameFeatures, pixNormSobel, relDist])

	def __getitem__(self, ind):
		return self.feat[ind]

	def __len__(self):
		return len(self.feat)

class FeatureGenPrevFrame:
	def __init__(self, trainNormSamples, numIntPcaComp, numShapePcaComp):
		self.numIntPcaComp = numIntPcaComp
		self.numShapePcaComp = numShapePcaComp
		self.pcaShape = converge.PcaNormShape(trainNormSamples)
		self.pcaInt = converge.PcaNormImageIntensity(trainNormSamples)

	def Gen(self, sample, model):
		eigenPcaInt = self.pcaInt.ProjectToPca(sample, model)[:self.numIntPcaComp]
		eigenShape = self.pcaShape.ProjectToPca(sample, model)[:self.numShapePcaComp]
		return np.concatenate([eigenPcaInt, eigenShape])

if __name__ == "__main__":
	pass

