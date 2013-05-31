
import supra, pickle, random
import numpy as np

def TrainTracker(trainNormSamples):
	cloudTracker = supra.SupraLayers(trainNormSamples)

	for sampleCount, sample in enumerate(trainNormSamples):
		print "train", sampleCount, len(trainNormSamples)
		cloudTracker.AddTraining(sample, 50)

	cloudTracker.PrepareModel()
	return cloudTracker

def TestTracker(cloudTracker, testNormSamples, log):
	testOffs = []
	testSampleStore = []
	testPredModels = []
	testModels = []
	trueModels = []
	for sampleCount, sample in enumerate(testNormSamples):
		print "test", sampleCount, len(testNormSamples)
		prevFrameFeat = cloudTracker.CalcPrevFrameFeatures(sample, sample.procShape)
		
		for layer in cloudTracker.layers:
			print layer.supportPixHalfWidth, layer.trainingOffset

		for count in range(10):
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
			testSampleStore.append(sample)

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
	for ptNum in range(testOffstestOffs.shape[1]):
		correlX = np.corrcoef(testOffs[:,ptNum,0], testPreds[:,ptNum,0])[0,1]
		correlY = np.corrcoef(testOffs[:,ptNum,1], testPreds[:,ptNum,1])[0,1]
		correl = 0.5*(correlX+correlY)
		correls.append(correl)
		#plt.plot(testOffs[:,ptNum,0], testPreds[:,ptNum,0],'x')
		#plt.plot(testOffs[:,ptNum,1], testPreds[:,ptNum,1],'x')
	#plt.savefig("correl.svg")
	
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
	medPredError = np.median(predErrors)
	offsetDist = np.array(offsetDist)

	print "correl",avCorrel
	print "signScore",avSignScore
	print "medPredError",medPredError

	#plt.plot(offsetDist[0,:], predErrors[0,:] ,'x')
	#plt.show()

	log.write(str(avCorrel)+","+str(avSignScore)+","+str(medPredError)+"\n")
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

	#DumpNormalisedImages(filteredSamples)

	#Reduce problem to n points
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
			cloudTracker = TrainTracker(trainNormSamples)
			print cloudTracker
			pickle.dump(cloudTracker, open("tracker.dat","wb"), protocol=-1)
			pickle.dump(testNormSamples, open("testNormSamples.dat","wb"), protocol=-1)
		else:
			cloudTracker = pickle.load(open("tracker.dat","rb"))
			testNormSamples = pickle.load(open("testNormSamples.dat","rb"))
			print cloudTracker

		#Run performance test
		TestTracker(cloudTracker, testNormSamples, log)

		#TestSingleExample(cloudTracker, testNormSamples, log)


