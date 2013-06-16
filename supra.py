
import numpy as np, pickle, random, pxutil, copy, math, sqlitedict, os, tempfile
import normalisedImage, normalisedImageOpt, simpleGbrt, supraFeatures
import skimage.color as col, skimage.feature as feature, skimage.filter as filt
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

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

	def __init__(self, ptNumIn, numPoints = 5, supportPixHalfWidthIn = 0.3, numSupportPix = 50):
		self.ptNum = ptNumIn
		self.supportPixOff = None
		self.supportPixOffSobel = None
		self.trainInt = []
		self.trainOffX, self.trainOffY = [], []
		self.regX, self.regY = None, None
		self.sobelKernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)
		self.sobelOffsets, halfWidth = normalisedImageOpt.CalcKernelOffsets(self.sobelKernel)
		self.featureMultiplex = simpleGbrt.FeatureGenTest()
		self.trainIntDb = None
		self.featureGen = supraFeatures.FeatureGen(numPoints, supportPixHalfWidthIn, numSupportPix)
		self.numPoints = numPoints

	def __del__(self):
		del self.trainIntDb
		try:
			if self.trainIntDbFina is not None:
				os.remove(self.trainIntDbFina)
		except:
			pass

	def AddTraining(self, sample, trainOffset, extraFeatures):
			
		xOff = trainOffset[self.ptNum][0]
		yOff = trainOffset[self.ptNum][1]

		if self.trainIntDb is None:
			self.trainIntDbFina = tempfile.mkstemp()[1]
			self.trainIntDb = sqlitedict.SqliteDict(self.trainIntDbFina, autocommit=True)

		self.featureGen.SetImage(sample)
		self.featureGen.SetModel(sample.procShape)
		self.featureGen.SetModelOffset(trainOffset)
		self.featureGen.SetShapeNoise(0.3)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(xOff, yOff)
		self.featureGen.Gen()
		feat = self.featureGen.GetGenFeat()
		featComp = np.concatenate((feat, extraFeatures))

		#self.trainInt.append(features)
		self.trainIntDb[str(len(self.trainOffX))] = featComp
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
		self.trainOffX, self.trainOffY = [], []

	def GetFeatureImportance(self):
		out = []
		for axis in self.axes:
			out.append(axis.GetFeatureImportance())
		return out

	def Predict(self, sample, model, prevFrameFeatures):
		self.featureGen.SetImage(sample)
		self.featureGen.SetModel(model)
		self.featureGen.ClearModelOffset()
		self.featureGen.SetShapeNoise(0.)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(0., 0.)
		feat = self.featureGen.Gen()
		featComp = np.concatenate((feat, prevFrameFeatures))

		#self.featureMultiplex.ClearFeatureSets()
		#self.featureMultiplex.AddFeatureSet(self.featureGen.GetGenFeat())

		totalx, totaly, weightx, weighty = 0., 0., 0., 0.
		for axis in self.axes:
			#pred = simpleGbrt.PredictGbrt(axis.reg, self.featureMultiplex)
			pred = axis.reg.predict([feat])[0]
			totalx += pred * axis.x
			totaly += pred * axis.y
			weightx += axis.x
			weighty += axis.y

		return totalx / weightx, totaly / weighty

	def SetFeatureMask(self, mask):
		self.featureGen.SetFeatureMask(mask)

	def GetFeatureList(self):
		return self.featureGen.GetFeatureList()

class SupraCloud():

	def __init__(self, supportPixHalfWidthIn = 0.3, trainingOffsetIn = 0.3, numPoints = 5):
		self.trainingOffset = trainingOffsetIn #Standard deviations
		self.supportPixHalfWidth = supportPixHalfWidthIn
		self.numIter = 2
		self.numPoints = numPoints

		self.trackers = []
		self.featureGen = []

		for i in range(self.numPoints):
			self.trackers.append(SupraAxisSet(i, self.numPoints, self.supportPixHalfWidth))

	def AddTraining(self, sample, numExamples, extraFeatures):

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

	def SetFeatureMasks(self, masks):
		for tracker, mask in zip(self.trackers, masks):
			tracker.SetFeatureMask(mask)

	def GetFeatureList(self):
		masks = []
		for tracker in self.trackers:
			masks.append(tracker.GetFeatureList())
		return masks

class SupraLayers:
	def __init__(self, trainNormSamples):
		self.featureGenPrevFrame = supraFeatures.FeatureGenPrevFrame(trainNormSamples, 20, 5)
		self.numPoints = trainNormSamples[0].NumPoints()
		if self.numPoints == 0:
			raise ValueError("Model must have non-zero number of points")
		for sample in trainNormSamples: 
			if sample.NumPoints() != self.numPoints:
				raise ValueError("Model must have consistent number of points")

		self.layers = [SupraCloud(0.3,0.2,self.numPoints),SupraCloud(0.3,0.05,self.numPoints)]

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

	def SetFeatureMasks(self, masks):
		for layer, masksIt in zip(self.layers, masks):
			layer.SetFeatureMasks(masksIt)

	def GetFeatureList(self):
		out = []
		for layer in self.layers:
			out.append(layer.GetFeatureList())
		return out

if __name__ == "__main__":
	pass

