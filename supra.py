
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
		self.reg = None
		self.x = axisXIn
		self.y = axisYIn
	
	def PrepareModel(self, features, offsets):
		if self.reg is not None:
			return 0

		self.reg = GradientBoostingRegressor()

		offsets = np.array(offsets)
		labels = offsets[:,0] * self.x + offsets[:,1] * self.y

		if not np.all(np.isfinite(labels)):
			raise Exception("Training labels contains non-finite value(s), either NaN or infinite")

		self.reg.fit(features, labels)

	def IsModelReady(self):
		return self.reg is not None

	def ClearModel(self):
		self.reg = None
	
	def GetFeatureImportance(self):
		return self.reg.feature_importances_

def ListCompare(la, lb):
	if la is None and lb is None: return True
	if la is None: return False
	if lb is None: return False
	if len(la) != len(lb): return False
	for a, b in zip(la, lb):
		if a != b: return False
	return True

class SupraAxisSet():

	def __init__(self, ptNumIn, numPoints = 5, supportPixHalfWidthIn = 0.3, numSupportPix = 50):
		self.ptNum = ptNumIn
		self.supportPixOff = None
		self.supportPixOffSobel = None
		self.trainInt = []
		self.trainOffX, self.trainOffY = [], []
		self.regX, self.regY = None, None
		self.sobelKernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.int32)
		#self.sobelOffsets, self.sobelCoeffs, halfWidth = normalisedImageOpt.CalcKernelOffsets(self.sobelKernel)
		#self.featureMultiplex = simpleGbrt.FeatureGenTest()
		self.trainIntDb = None
		self.numSupportPix = numSupportPix
		self.featureGen = supraFeatures.FeatureGen(numPoints, supportPixHalfWidthIn, numSupportPix, 1)
		self.numPoints = numPoints
		self.featureMask = None
		self.axes = None

	def __del__(self):
		del self.trainIntDb
		try:
			if self.trainIntDbFina is not None:
				os.remove(self.trainIntDbFina)
		except:
			pass

	def SetParameters(self, params):
		pass

	def IsModelReady(self):
		if self.axes is None:
			return False
		countUnready = 0
		for axis in self.axes:
			if not axis.IsModelReady():
				countUnready += 1
		if countUnready > 0: return False
		return True

	def AddTraining(self, sample, trainOffset, extraFeatures):

		#Check at least one axis requires data
		if self.IsModelReady(): return 0

		xOff = trainOffset[self.ptNum][0]
		yOff = trainOffset[self.ptNum][1]

		if self.trainIntDb is None:
			self.trainIntDbFina = tempfile.mkstemp()[1]
			self.trainIntDb = sqlitedict.SqliteDict(self.trainIntDbFina, autocommit=True)

		self.featureGen.SetImage(sample)
		self.featureGen.SetModel(np.array(sample.procShape))
		self.featureGen.SetModelOffset(trainOffset)
		self.featureGen.SetShapeNoise(0.6)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(xOff, yOff)
		self.featureGen.Gen()
		feat = self.featureGen.GetGenFeat()

		if feat.shape[0] != len(self.featureMask):
			print "Warning: Generated features have incorrect number of components, got "+\
				str(feat.shape[0])+", received "+str(len(self.featureMask))

		featComp = np.concatenate((feat, extraFeatures))

		if not np.all(np.isfinite(featComp)):
			for i, comp in enumerate(featComp):
				print comp, math.isinf(comp), math.isnan(comp),
				if i < len(self.featureMask):
					print self.featureMask[i]
				else:
					print ""

			raise Exception("Training data contains non-finite value(s), either NaN or infinite (1)")

		#self.trainInt.append(features)
		self.trainIntDb[str(len(self.trainOffX))] = featComp
		self.trainOffX.append(xOff)
		self.trainOffY.append(yOff)
		return 1

	def PrepareModel(self):

		#Check at least one axis requires data
		if self.IsModelReady(): return 0

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

		if not np.all(np.isfinite(self.trainInt)):
			raise Exception("Training data contains non-finite value(s), either NaN or infinite (2)")

		for axis in self.axes:
			axis.PrepareModel(self.trainInt, trainOff)

		self.trainInt = None
		return 1

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
		self.featureGen.SetModel(np.array(model))
		self.featureGen.ClearModelOffset()
		self.featureGen.SetShapeNoise(0.)
		self.featureGen.SetPointNum(self.ptNum)
		self.featureGen.SetOffset(0., 0.)
		self.featureGen.Gen()

		feat = self.featureGen.GetGenFeat()
		featComp = np.concatenate((feat, prevFrameFeatures))
		#self.featureMultiplex.ClearFeatureSets()
		#self.featureMultiplex.AddFeatureSet(self.featureGen.GetGenFeat())

		totalx, totaly, weightx, weighty = 0., 0., 0., 0.
		for axis in self.axes:
			#pred = simpleGbrt.PredictGbrt(axis.reg, self.featureMultiplex)
			pred = axis.reg.predict([featComp])[0]
			totalx += pred * axis.x
			totaly += pred * axis.y
			weightx += axis.x
			weighty += axis.y

		return totalx / weightx, totaly / weighty

	def SetFeatureMask(self, mask):
		#if not ListCompare(mask, self.featureMask):#Hack, this should be done selectively
		if 1:
			self.featureMask = mask
			self.featureGen.SetFeatureMask(mask)
			self.ClearModels()
			return 1
		return 0

	def GetFeatureList(self):
		return self.featureGen.GetFeatureList()

	def ClearModels(self):
		if self.axes is not None: 
			for axis in self.axes:
				axis.ClearModel()

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

	def SetParameters(self, params):
		if 'trainingOffset' in params:
			#if self.trainingOffset != params['trainingOffset']: #Hack to check effect
			#	self.ClearModels()
			self.trainingOffset = params['trainingOffset']
			print "trainingOffset=", self.trainingOffset

	def AddTraining(self, sample, numExamples, extraFeatures):

		mod = 0
		for sampleCount in range(numExamples):
			perturb = []
			for num in range(sample.NumPoints()):
				perturb.append((np.random.normal(scale=self.trainingOffset),\
					np.random.normal(scale=self.trainingOffset)))

			for count, tracker in enumerate(self.trackers):
				mod += tracker.AddTraining(sample, perturb, extraFeatures)
		if mod == 0:
			raise Exception("No training added")

	def PrepareModel(self):
		mod = 0
		for tracker in self.trackers:
			mod += tracker.PrepareModel()
		if mod == 0:
			raise Exception("No change in model")
	
	def ClearTraining(self):
		for tracker in self.trackers:
			tracker.ClearTraining()

	def IsModelReady(self):
		for tracker in self.trackers:
			if not tracker.IsModelReady(): return False
		return True

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
		changed = 0
		if len(self.trackers) != len(masks):
			raise Exception("Number of trackers is incorrect")
		for tracker, mask in zip(self.trackers, masks):
			changed += tracker.SetFeatureMask(mask)
		return changed

	def GetFeatureList(self):
		masks = []
		for tracker in self.trackers:
			masks.append(tracker.GetFeatureList())
		return masks

	def ClearModels(self):
		for tracker in self.trackers:
			tracker.ClearModels()

class SupraLayers:
	def __init__(self, trainNormSamples):

		self.numPoints = trainNormSamples[0].NumPoints()
		self.featureGenPrevFrame = supraFeatures.FeatureGenPrevFrame(trainNormSamples, 20, self.numPoints)
		if self.numPoints == 0:
			raise ValueError("Model must have non-zero number of points")
		for sample in trainNormSamples: 
			if sample.NumPoints() != self.numPoints:
				raise ValueError("Model must have consistent number of points")

		self.layers = [SupraCloud(0.3,0.2,self.numPoints),SupraCloud(0.3,0.05,self.numPoints)]

	def SetParameters(self, params):
		if params is None:
			raise Exception("Invalid input")
		if len(params) != len(self.layers):
			raise Exception("Incorrect number of param layers")
		for player, layer in zip(params, self.layers):
			layer.SetParameters(player)

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

	def IsModelReady(self):
		for layer in self.layers:
			if not layer.IsModelReady(): return False
		return True

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
		changed = 0
		if len(self.layers) != len(masks):
			raise Exception("Number of layers is incorrect")
		for layer, masksIt in zip(self.layers, masks):
			changed += layer.SetFeatureMasks(masksIt)
		return changed

	def GetFeatureList(self):
		out = []
		for layer in self.layers:
			out.append(layer.GetFeatureList())
		return out

	def ClearModels(self):
		for layer in self.layers:
			layer.ClearModels()


if __name__ == "__main__":
	pass

