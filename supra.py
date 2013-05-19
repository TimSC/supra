
import numpy as np, pickle, random, pxutil, copy, math, normalisedImage
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
		self.axisx = axisXIn
		self.axisy = axisYIn
		self.model = None
	
	def PrepareModel(self, features, offsets):
		self.model = GradientBoostingRegressor()
		labels = offsets[:,0] * self.axisx + offsets[:,1] * self.axisy

		self.model.fit(features, labels)

class SupraAxisSet():

	def __init__(self, ptNumIn):
		self.supportPixHalfWidth = 0.3
		self.supportPixNum = 50
		self.supportPixOff = None
		self.regAxis = None
		self.ptNum = ptNumIn
		self.trainFeatures = []
		self.trainOffsets = []
		self.holisticFeat = None

	def AddTraining(self, sample, trainOffset):

		#Initialise support pixel locations
		if self.supportPixOff is None:
			self.supportPixOff = np.random.uniform(low=-self.supportPixHalfWidth, \
				high=self.supportPixHalfWidth, size=(self.supportPixNum, 2))

		#Extract support pixel intensities
		trainX = trainOffset[self.ptNum][0]
		trainY = trainOffset[self.ptNum][1]
		ptX = sample.procShape[self.ptNum][0]
		ptY = sample.procShape[self.ptNum][1]
		pix = ExtractSupportIntensity(sample, self.supportPixOff, ptX, ptY, \
			trainX, trainY)
		pixConv = []
		for px in pix:
			pixConv.extend(ColConv(px))
		pixNorm = np.array(pixConv)
		pixNorm -= pixNorm.mean()

		#Extract texture using HOG #SIMP
		#localPatch = col.rgb2grey(normalisedImage.ExtractPatch(sample, self.ptNum, trainX, trainY))
		#hog = feature.hog(localPatch)

		feat = np.concatenate([pixNorm]) #SIMP
		self.trainFeatures.append(feat)
		self.trainOffsets.append(trainOffset)

	def AddHolisticFeatures(self, feat):
		self.holisticFeat = feat

	def PrepareModel(self):
		combinedTrainingData = np.array(self.trainFeatures)
		if self.holisticFeat is not None:
			combinedTrainingData = np.hstack([combinedTrainingData, self.holisticFeat])

		self.regAxis = []
		self.regAxis.append(SupraAxis(math.cos(math.radians(0.)), math.sin(math.radians(0.))))
		#self.regAxis.append(SupraAxis(math.cos(math.radians(45.)), math.sin(math.radians(45.)))) #SIMP
		self.regAxis.append(SupraAxis(math.cos(math.radians(90.)), math.sin(math.radians(90.))))
		#self.regAxis.append(SupraAxis(math.cos(math.radians(135.)), math.sin(math.radians(135.)))) #SIMP

		for axis in self.regAxis:
			axis.PrepareModel(combinedTrainingData, np.array(self.trainOffsets)[:,self.ptNum,:])

	def Predict(self, sample, model, prevFrameFeatures):

		ptX = model[self.ptNum,0]
		ptY = model[self.ptNum,1]

		#Extract sparse pixel intensities
		pix = ExtractSupportIntensity(sample, self.supportPixOff, ptX, ptY, 0., 0.)
		pixConv = []
		for px in pix:
			pixConv.extend(ColConv(px))
		pixNorm = np.array(pixConv)
		pixNorm -= pixNorm.mean()

		#Extract texture using HOG #SIMP
		#localPatch = col.rgb2grey(normalisedImage.ExtractPatchAtImg(sample, ptX, ptY))
		#hog = feature.hog(localPatch)
		
		combinedTrainingData = np.concatenate([pixNorm]) #SIMP
		if prevFrameFeatures is not None:
			combinedTrainingData = np.hstack([combinedTrainingData, prevFrameFeatures])

		#Get prediction from axis
		totalX, totalY, weightX, weightY = 0., 0., 0., 0.
		for axis in self.regAxis:
			pr = axis.model.predict(combinedTrainingData)
			totalX += pr * axis.axisx
			totalY += pr * axis.axisy
			weightX += axis.axisx
			weightY += axis.axisy
		predX = totalX / weightX
		predY = totalY / weightY

		return (predX, predY)

class SupraCloud():

	def __init__(self):
		self.imgsSparseInt = []
		self.flattenedShapes = []
		self.sampleIndex = []
		self.trackers = None
		self.sparseAppTemplate = None
		self.numShapeComp = 5
		self.numAppComp = 20
		self.numIterations = 1 #SIMP

	def AddTraining(self, sample, numExamples):

		#Store shape
		sampleModel = np.array(sample.procShape)
		sampleModel = sampleModel.reshape(sampleModel.size)
		self.flattenedShapes.append(sampleModel)

		#Store intensity
		if self.sparseAppTemplate is None:
			self.sparseAppTemplate = np.random.uniform(low=-0.3, \
					high=0.3, size=(50, 2))
		self.imgsSparseInt.append(self.ExtractFeatures(sample, sample.procShape))

		#Init trackers, if necessary
		if self.trackers is None:
			self.trackers = [SupraAxisSet(ptNum) for ptNum in range(len(sample.procShape))]

		#Perturb tracker positions
		count = 0
		while count < numExamples:
			
			trainOffset = []
			for pt in range(sample.procShape.shape[0]):
				x = np.random.normal(scale=0.3)
				y = 0. #np.random.normal(scale=0.3)#SIMP
				trainOffset.append((x,y))

			for tr in self.trackers:
				tr.AddTraining(sample, trainOffset)
				
			self.sampleIndex.append(len(self.flattenedShapes)-1)

			count += 1

	def PrepareModel(self):
		
		#Prepare PCA of shape
		shapesArr = np.array(self.flattenedShapes)
		self.meanShape = shapesArr.mean(axis=0)
		meanShapeCent = shapesArr - self.meanShape
	
		self.shapeu, self.shapes, self.shapev = np.linalg.svd(meanShapeCent, full_matrices=False)
		self.shapeS = np.zeros((self.shapeu.shape[0], self.shapev.shape[0]))
		self.shapeS[:self.shapes.shape[0], :self.shapes.shape[0]] = np.diag(self.shapes)
		#reconstructed = np.dot(self.shapeu, np.dot(self.shapeS, self.shapev))

		#Project shape samples into PCA space
		shapeEigVecs = []
		for shape in self.flattenedShapes:
			shapeEigVec = self.ProjectShapeToPca(shape)
			shapeEigVecs.append(shapeEigVec)
		shapeEigVecs = np.array(shapeEigVecs)[:,:self.numShapeComp]
		shapeEigVecs = shapeEigVecs[self.sampleIndex,:]

		#Prepare PCA of intensity of sparse image
		imgsSparseIntArr = np.array(self.imgsSparseInt)
		self.meanInt = imgsSparseIntArr.mean(axis=0)
		imgsSparseIntCent = imgsSparseIntArr - self.meanInt

		self.appu, self.apps, self.appv = np.linalg.svd(imgsSparseIntCent, full_matrices=False)
		self.appS = np.zeros((self.appu.shape[0], self.appv.shape[0]))
		self.appS[:self.apps.shape[0], :self.apps.shape[0]] = np.diag(self.apps)
		#reconstructed = np.dot(self.u, np.dot(self.S, self.v))

		#Project appearance samples into PCA space
		appEigVecs = []
		for app in self.imgsSparseInt:
			appEigVec = self.ProjectAppearanceToPca(app)
			appEigVecs.append(appEigVec)
		appEigVecs = np.array(appEigVecs)[:,:self.numAppComp]
		appEigVecs = appEigVecs[self.sampleIndex,:]

		#for tr in self.trackers:
		#	tr.AddHolisticFeatures(np.hstack([shapeEigVecs, appEigVecs])) #SIMP
	
		for tr in self.trackers:
			tr.PrepareModel()

	def ExtractFeatures(self, sample, model):
		imgSparseInt = []
		for pt in range(len(model)):
			ptX = model[pt][0]
			ptY = model[pt][1]
			pix = ExtractSupportIntensity(sample, self.sparseAppTemplate, ptX, ptY, 0., 0.)
			pixGrey = [pxutil.ToGrey(p) for p in pix]
			imgSparseInt.extend(pixGrey)		
		return imgSparseInt

	def CalcPrevFrameFeatures(self, sample, model):
		return None
		flatShape = np.array(model).reshape(model.size)
		shapeEigVecs = self.ProjectShapeToPca(flatShape)
		
		app = self.ExtractFeatures(sample, model)
		appEigVecs = self.ProjectAppearanceToPca(app)
		return np.hstack([shapeEigVecs[:self.numShapeComp], appEigVecs[:self.numAppComp]])

	def Predict(self, sample, model, prevFrameFeatures):
		currentModel = copy.deepcopy(model)
		
		for numInter in range(self.numIterations):
			for num, tracker in enumerate(self.trackers):
				pred = tracker.Predict(sample, model, prevFrameFeatures)
				return pred #SIMP
				currentModel[num,0] -= pred[0]
				currentModel[num,1] -= pred[1]
		return currentModel

	def ProjectAppearanceToPca(self, sample):
		centred = sample - self.meanInt
		return np.dot(centred, self.appv.transpose()) / self.apps

	def ProjectShapeToPca(self, sample):
		centred = sample - self.meanShape
		return np.dot(centred, self.shapev.transpose()) / self.shapes

def TrainTracker(trainNormSamples):
	cloudTracker = SupraCloud()
	
	print "Generating synthetic training data"
	for count, sample in enumerate(trainNormSamples):
		print count
		cloudTracker.AddTraining(sample, 5)
	
	print "Preparing Model"
	cloudTracker.PrepareModel()
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

	#Reduce problem to one point
	for sample in filteredSamples:
		sample.procShape = sample.procShape[0:1,:]

	fi = open("log.txt","wt")

	while 1:
		print "Filtered to",len(filteredSamples),"of",len(normalisedSamples),"samples"
		halfInd = len(filteredSamples)/2
		random.shuffle(filteredSamples)
		trainNormSamples = filteredSamples[:halfInd]
		testNormSamples = filteredSamples[halfInd:]

		if 1:
			cloudTracker = TrainTracker(trainNormSamples)
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
		fi.write(str(correl)+"\n")
		fi.flush()
		#plt.plot(testVal, testErr, 'bx')
		#plt.show()

