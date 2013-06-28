
import supra, pickle, random, normalisedImage, normalisedImageOpt, copy, sys, traceback
import numpy as np
from multiprocessing import Pool, cpu_count

def PredictPoint(trackerLayer, sample, model, prevFrameFeatures, trLogPointIn = None):
	currentModel = np.array(copy.deepcopy(model))
	trLogPoint = []

	for iterNum in range(trackerLayer.numIter):
		for ptNum, tracker in enumerate(trackerLayer.trackers):
			while ptNum >= len(trLogPoint):
				trLogPoint.append([])
			trLogPoint[ptNum].append(currentModel[ptNum,:])

			pred = tracker.Predict(sample, currentModel, prevFrameFeatures)
			currentModel[ptNum,:] -= pred

	return currentModel, trLogPoint

def PredictLayers(tracker, sample, model, prevFrameFeatures, trLogIn = None):
	currentModel = np.array(copy.deepcopy(model))
	trLogOut = []
	if trLogIn is None:
		trLogIn = [[] for layer in tracker.layers]

	for layerNum, (layer, trLogPoint) in enumerate(zip(tracker.layers, trLogIn)):
		currentModel, trLogPoint = PredictPoint(layer, sample, currentModel, prevFrameFeatures, trLogPoint)
		trLogOut.append(trLogPoint)
	return currentModel, trLogOut

class TrainEval:
	def __init__(self, trainNormSamples, tracker = None):
		if tracker is None:
			self.cloudTracker = supra.SupraLayers(trainNormSamples)
		else:
			self.cloudTracker = tracker
		self.masks = self.cloudTracker.GetFeatureList()
		self.fullMasks = copy.deepcopy(self.masks)
		self.testOffStore = None

	def Train(self, trainNormSamples, numTrainOffsets = 10):

		if 0:
			filtMasks = []
			for layer in self.masks:
				l = []
				for tr in layer:
					t = []
					for f in tr:
						if f[:3] != "hog":
							t.append(f)
					l.append(t)
				filtMasks.append(l)
		else:
			filtMasks = self.masks

		self.cloudTracker.SetFeatureMasks(filtMasks)

		for sampleCount, sample in enumerate(trainNormSamples):
			print "train", sampleCount, len(trainNormSamples)
			self.cloudTracker.AddTraining(sample, numTrainOffsets) #35

		self.cloudTracker.PrepareModel()
		self.cloudTracker.ClearTraining()

	def SetFeatureMasks(self, masks):
		self.masks = masks

	def InitRandomMask(self, frac=0.1):
		for layer in self.masks:
			for trackerNum, tracker in enumerate(layer):
				filt = []				
				for featComp in tracker:
					if random.random() < frac:
						filt.append(featComp)
				layer[trackerNum] = filt

	def Test(self, testNormSamples, numTestOffsets = 10, log = None):
		#testOffsCollect = []
		sampleInfo = []
		testPredModels = []
		testModels = []
		trueModels = []

		if self.testOffStore is None:
			self.testOffStore = []
			for sampleCount, sample in enumerate(testNormSamples):
				testOff = []
				for count in range(numTestOffsets):
					x = np.random.normal(scale=0.3)
					y = np.random.normal(scale=0.3)
					testOff.append((x, y))
				self.testOffStore.append(testOff)

		for sampleCount, (sample, testOffs) in enumerate(zip(testNormSamples, self.testOffStore)):
			print "test", sampleCount, len(testNormSamples), sample.info['roiId']
			prevFrameFeat = self.cloudTracker.CalcPrevFrameFeatures(sample, sample.GetProcrustesNormedModel())
		
			for layer in self.cloudTracker.layers:
				print layer.supportPixHalfWidth, layer.trainingOffset

			for count, testOff in enumerate(testOffs):
				#Purturb positions for testing
				testPos = []
				for pt in sample.GetProcrustesNormedModel():
					testPos.append((pt[0] + testOff[0], pt[1] + testOff[1]))

				#Make predicton
				predModel, trackLog = PredictLayers(self.cloudTracker, sample, testPos, prevFrameFeat)

				#Store result
				#testOffsCollect.append(testOff)
				testPredModels.append(predModel)
				testModels.append(testPos)
				trueModels.append(sample.GetProcrustesNormedModel())
				sampleInfo.append(sample.info)


		#Calculate performance
		testOffsCollect = np.array(self.testOffStore)
		testPredModels = np.array(testPredModels)
		testModels = np.array(testModels)
		trueModels = np.array(trueModels)

		#Calculate relative movement of tracker
		testPreds = []
		for sampleNum in range(testOffsCollect.shape[0]):
			diff = []
			for ptNum in range(testOffsCollect.shape[1]):
				diff.append((testModels[sampleNum,ptNum,0] - testPredModels[sampleNum,ptNum,0], \
					testModels[sampleNum,ptNum,1] - testPredModels[sampleNum,ptNum,1]))
			testPreds.append(diff)
		testPreds = np.array(testPreds)

		#Calculate performance metrics
		correls, signScores = [], []
		for ptNum in range(testOffsCollect.shape[1]):
			correlX = np.corrcoef(testOffsCollect[:,ptNum,0], testPreds[:,ptNum,0])[0,1]
			correlY = np.corrcoef(testOffsCollect[:,ptNum,1], testPreds[:,ptNum,1])[0,1]
			correl = 0.5*(correlX+correlY)
			correls.append(correl)
			#plt.plot(testOffsCollect[:,ptNum,0], testPreds[:,ptNum,0],'x')
			#plt.plot(testOffsCollect[:,ptNum,1], testPreds[:,ptNum,1],'x')
		#plt.savefig("correl.svg")
	
		for ptNum in range(testOffsCollect.shape[1]):
			signX = supra.SignAgreement(testOffsCollect[:,ptNum,0], testPreds[:,ptNum,0])
			signY = supra.SignAgreement(testOffsCollect[:,ptNum,1], testPreds[:,ptNum,1])
			signScore = 0.5 * (signX + signY)
			signScores.append(signScore)

		#Calculate prediction error	
		predErrors, offsetDist = [], []
		for ptNum in range(testOffsCollect.shape[1]):
			errX = trueModels[:,ptNum,0] - testPredModels[:,ptNum,0]
			errY = trueModels[:,ptNum,1] - testPredModels[:,ptNum,1]
			offset = np.power(np.power(testOffsCollect[:,ptNum,0],2.)+np.power(testOffsCollect[:,ptNum,1],2.),0.5)
			err = np.power(np.power(errX,2.)+np.power(errY,2.),0.5)
			predErrors.append(err)
			offsetDist.append(offset)

		#Get average performance
		avCorrel = np.array(correls).mean()
		avSignScore = np.array(signScores).mean()
		predErrors = np.array(predErrors)
		medPredError = np.median(predErrors)
		offsetDist = np.array(offsetDist)

		print "correl",avCorrel
		print "signScore",avSignScore
		print "medPredError",medPredError

		#Calc sample specific error
		roiDict = {}
		for info, errNum in zip(sampleInfo, range(predErrors.shape[1])):
			roiId = info['roiId']
			if roiId not in roiDict:
				roiDict[roiId] = []
			roiDict[roiId].append(predErrors[:, errNum])
		#pickle.dump(roiDict, open("roiDict.dat", "wb"), protocol = -1)

		#plt.plot(offsetDist[0,:], predErrorsArr[0,:] ,'x')
		#plt.show()

		if log is not None:
			log.write(str(avCorrel)+","+str(avSignScore)+","+str(medPredError)+"\n")
			log.flush()

		return {'avCorrel':avCorrel, 'avSignScore': avSignScore, 'medPredError': medPredError, 'model': self.cloudTracker}

def EvalTrackerConfig(args):
	try:
		currentConfig = args[0]
		trainNormSamples = args[1]
		testNormSamples = args[2]
		testMasks = args[3]

		currentConfig.SetFeatureMasks(testMasks)
		currentConfig.Train(trainNormSamples, 1)#Hack
		perf = currentConfig.Test(testNormSamples, 1)#Hack
		del currentConfig
	except Exception as err:
		print err
		traceback.print_exc(file=sys.stdout)
		return None
	return perf

class FeatureSelection:
	def __init__(self, normalisedSamples):
		self.currentConfig = None
		self.log = open("log.txt","wt")
		self.metric = 'medPredError'
		self.SplitSamples(normalisedSamples)
		self.tracker = supra.SupraLayers(self.trainNormSamples)
		self.currentMask = None

	def SplitSamples(self, normalisedSamples):
		halfInd = len(filteredSamples)/2
		random.shuffle(filteredSamples)
		self.trainNormSamples = filteredSamples[:halfInd]
		self.testNormSamples = filteredSamples[halfInd:]

		mirImgs = [normalisedImage.HorizontalMirrorNormalisedImage(img,[1,0,2,4,3]) for img in self.trainNormSamples]
		self.trainNormSamples.extend(mirImgs)

	def EvaluateForwardSteps(self, numTests=None):
		if self.currentConfig == None:
			self.currentConfig = TrainEval(self.trainNormSamples, copy.deepcopy(self.tracker))
		else:
			self.currentConfig.cloudTracker = self.tracker

		if self.currentMask == None:
			self.currentConfig.InitRandomMask()
			self.currentMask = self.currentConfig.masks
		else:
			self.currentConfig.SetFeatureMasks(self.currentMask)
		
		#Plan which componenets to test
		componentsToTest = []
		for layerNum, (layers, fullMaskLayers) in enumerate(zip(self.currentMask,
				self.currentConfig.fullMasks)):
			for trackerNum, (mask, fullMask) in enumerate(zip(layers, fullMaskLayers)):
				for component in fullMask:
					if component not in mask:
						componentsToTest.append((layerNum, trackerNum, component, "Forward"))

		print "Num components to test in forward step", len(componentsToTest)

		if numTests is None:
			numTests = len(componentsToTest)
		componentsToTest = random.sample(componentsToTest, numTests)

		print "Using a sample of size", len(componentsToTest)

		#Evaluate each component
		testArgList = []
		for test in componentsToTest:
			testLayer = test[0]
			testTracker = test[1]
			testComponent = test[2]

			#Create temporary mask
			testMasks = copy.deepcopy(self.currentMask)
			testMasks[testLayer][testTracker].append(testComponent)

			testArgList.append((self.currentConfig, self.trainNormSamples, self.testNormSamples, testMasks))

		print "Forward step evaluate"
		pool = Pool(processes=cpu_count())
		evalPerfs = pool.map(EvalTrackerConfig, testArgList)
		pool.close()
		pool.join()

		testPerfs = []
		for perf, test, testArgs in zip(evalPerfs, componentsToTest, testArgList):
			model = perf['model']
			del perf['model']

			testPerfs.append((perf[self.metric], perf, test, testArgs, model))
			self.log.write(str(test)+str(perf)+"\n")
			self.log.flush()

		testPerfs.sort()
		
		return testPerfs

	def EvaluateBackwardSteps(self, numTests=None):

		if self.currentConfig == None:
			self.currentConfig = TrainEval(self.trainNormSamples, copy.deepcopy(self.tracker))
		if self.currentMask == None:
			self.currentConfig.InitRandomMask()
			self.currentMask = self.currentConfig.masks
		else:
			self.currentConfig.SetFeatureMasks(self.currentMask)
		
		#Plan which componenets to test
		componentsToTest = []
		for layerNum, layers in enumerate(self.currentMask):
			for trackerNum, mask in enumerate(layers):
				for component in mask:
					componentsToTest.append((layerNum, trackerNum, component, "Backward"))

		print "Num components to test in backward step", len(componentsToTest)

		if numTests is None:
			numTests = len(componentsToTest)
		componentsToTest = random.sample(componentsToTest, numTests)

		print "Using a sample of size", len(componentsToTest)

		#Evaluate each component
		testArgList = []
		for test in componentsToTest:
			testLayer = test[0]
			testTracker = test[1]
			testComponent = test[2]

			#Create temporary mask
			testMasks = copy.deepcopy(self.currentMask)
			testMasks[testLayer][testTracker].append(testComponent)

			testArgList.append((self.currentConfig, self.trainNormSamples, self.testNormSamples, testMasks))

		pool = Pool(processes=cpu_count())
		evalPerfs = pool.map(EvalTrackerConfig, testArgList)
		pool.close()
		pool.join()

		testPerfs = []
		for perf, test, testArgs in zip(evalPerfs, componentsToTest, testArgList):
			model = perf['model']
			del perf['model']

			testPerfs.append((perf[self.metric], perf, test, testArgs, model))
			self.log.write(str(test)+str(perf)+"\n")
			self.log.flush()

		testPerfs.sort()
		
		return testPerfs

	def SetFeatureMasks(self, masks):
		self.currentMask = masks

	def ClearCurrentModel(self):
		self.currentConfig = None

def EvalSingleConfig(filteredSamples):
	
	#Reduce problem to n points
	#for sample in filteredSamples:
	#	sample.procShape = sample.GetProcrustesNormedModel()[0:1,:]

	log = open("log.txt","wt")

	while 1:
		halfInd = len(filteredSamples)/2
		random.shuffle(filteredSamples)
		trainNormSamples = filteredSamples[:halfInd]
		testNormSamples = filteredSamples[halfInd:]
		trainTracker = TrainEval(trainNormSamples)

		if 1:
			#Reflect images to increase training data
			mirImgs = [normalisedImage.HorizontalMirrorNormalisedImage(img,[1,0,2,4,3]) for img in trainNormSamples]
			trainNormSamples.extend(mirImgs)
			#trainNormSamples = mirImgs

			#Create and train tracker
			#trainTracker.InitRandomMask()
			cloudTracker = trainTracker.Train(trainNormSamples, 10)
			
			cloudTracker = trainTracker.cloudTracker
			print cloudTracker
			pickle.dump(cloudTracker, open("tracker.dat","wb"), protocol=-1)
			pickle.dump(testNormSamples, open("testNormSamples.dat","wb"), protocol=-1)
		else:
			cloudTracker = pickle.load(open("tracker.dat","rb"))
			testNormSamples = pickle.load(open("testNormSamples.dat","rb"))
			print cloudTracker
			trainTracker.cloudTracker = cloudTracker

		#Run performance test
		trainTracker.Test(testNormSamples, 10, log)

def FeatureSelectRunScript(filteredSamples):

	featureSelection = FeatureSelection(filteredSamples)
	pickle.dump(featureSelection.tracker, open("fsmodel.dat", "wb"), protocol = -1)
	fslog = open("fslog.txt","wt")

	running = True
	count = 0
	currentModel = featureSelection.tracker

	while running:
		featureSelection.SplitSamples(filteredSamples)
		featureSelection.tracker = currentModel
	
		perfs = featureSelection.EvaluateForwardSteps(4)#Hack
		#perfs2 = featureSelection.EvaluateBackwardSteps(16)#Hack
		#perfs.extend(perfs2)#Hack
		perfs.sort()

		#Find best feature
		if len(perfs) > 0:
			bestMasks = perfs[0]
			featureSelection.SetFeatureMasks(bestMasks[3][3])
			currentModel = bestMasks[4]

			pickle.dump(bestMasks[3][3], open("masks"+str(count)+".dat", "wt"), protocol = 0)
			pickle.dump(currentModel, open("model"+str(count)+".dat", "wb"), protocol = -1)
			pickle.dump([x[:3] for x in perfs], open("iter"+str(count)+".dat", "wt"), protocol = 0)
			fslog.write(str(bestMasks[:3])+"\n")
			fslog.flush()
			count += 1
		else:
			running = False


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
	print "Filtered to",len(filteredSamples),"of",len(normalisedSamples),"samples"

	#DumpNormalisedImages(filteredSamples)
	if len(sys.argv) >= 2 and sys.argv[1] == "fs":
		FeatureSelectRunScript(filteredSamples)
	else:
		EvalSingleConfig(filteredSamples)

