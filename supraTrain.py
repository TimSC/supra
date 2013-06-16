
import supra, pickle, random, normalisedImage, normalisedImageOpt, copy
import numpy as np

class TrainEval:
	def __init__(self, trainNormSamples):
		self.cloudTracker = supra.SupraLayers(trainNormSamples)
		self.masks = self.cloudTracker.GetFeatureList()
		self.fullMasks = copy.deepcopy(self.masks)

	def Train(self, trainNormSamples):

		self.cloudTracker.SetFeatureMasks(self.masks)

		for sampleCount, sample in enumerate(trainNormSamples):
			print "train", sampleCount, len(trainNormSamples)
			self.cloudTracker.AddTraining(sample, 2) #35

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

	def Test(self, testNormSamples, log = None):
		testOffs = []
		sampleInfo = []
		testPredModels = []
		testModels = []
		trueModels = []
		for sampleCount, sample in enumerate(testNormSamples):
			print "test", sampleCount, len(testNormSamples), sample.info['roiId']
			prevFrameFeat = self.cloudTracker.CalcPrevFrameFeatures(sample, sample.GetProcrustesNormedModel())
		
			for layer in self.cloudTracker.layers:
				print layer.supportPixHalfWidth, layer.trainingOffset

			for count in range(10):
				#Purturb positions for testing
				testPos = []
				testOff = []
				for pt in sample.GetProcrustesNormedModel():
					x = np.random.normal(scale=0.3)
					y = np.random.normal(scale=0.3)
					testOff.append((x, y))
					testPos.append((pt[0] + x, pt[1] + y))

				#Make predicton
				predModel = self.cloudTracker.Predict(sample, testPos, prevFrameFeat)

				#Store result
				testOffs.append(testOff)
				testPredModels.append(predModel)
				testModels.append(testPos)
				trueModels.append(sample.GetProcrustesNormedModel())
				sampleInfo.append(sample.info)


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
		#plt.savefig("correl.svg")
	
		for ptNum in range(testOffs.shape[1]):
			signX = supra.SignAgreement(testOffs[:,ptNum,0], testPreds[:,ptNum,0])
			signY = supra.SignAgreement(testOffs[:,ptNum,1], testPreds[:,ptNum,1])
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

		return {'avCorrel':avCorrel, 'avSignScore': avSignScore, 'medPredError': medPredError}

class FeatureSelection:
	def __init__(self):
		self.currentConfig = None
		self.log = open("log.txt","wt")

	def SplitSamples(self, normalisedSamples):
		halfInd = len(filteredSamples)/2
		random.shuffle(filteredSamples)
		self.trainNormSamples = filteredSamples[:halfInd]
		self.testNormSamples = filteredSamples[halfInd:]

		mirImgs = [normalisedImage.HorizontalMirrorNormalisedImage(img,[1,0,2,4,3]) for img in self.trainNormSamples]
		self.trainNormSamples.extend(mirImgs)

	def EvaluateForwardSteps(self):
		if self.currentConfig == None:
			self.currentConfig = TrainEval(self.trainNormSamples)
			self.currentConfig.InitRandomMask()
			self.currentMask = self.currentConfig.masks
		
		#Plan which componenets to test
		componentsToTest = []
		for layerNum, (layers, fullMaskLayers) in enumerate(zip(self.currentMask,
				self.currentConfig.fullMasks)):
			for trackerNum, (mask, fullMask) in enumerate(zip(layers, fullMaskLayers)):
				for component in fullMask:
					if component not in mask:
						componentsToTest.append((layerNum, trackerNum, component))

		print "Num components to test in forward step", len(componentsToTest)

		#Evaluate each component
		testPerfs = []
		for test in componentsToTest:
			testLayer = test[0]
			testTracker = test[1]
			testComponent = test[2]

			#Create temporary mask
			self.testMasks = copy.deepcopy(self.currentMask)
			self.testMasks[testLayer][testTracker].append(testComponent)

			#Evaluate performance
			self.currentConfig.SetFeatureMasks(self.testMasks)
			cloudTracker = self.currentConfig.Train(self.trainNormSamples)
			perf = self.currentConfig.Test(self.testNormSamples)

			#Store result
			testPerfs.append((perf, self.testMasks))
			self.log.write(str(test)+str(perf)+"\n")
			self.log.flush()

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
			cloudTracker = trainTracker.Train(trainNormSamples)
			
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
		trainTracker.Test(testNormSamples, log)

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

	featureSelection = FeatureSelection()
	featureSelection.SplitSamples(filteredSamples)
	featureSelection.EvaluateForwardSteps()

