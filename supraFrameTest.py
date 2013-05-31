
import supra, pickle, normalisedImage
import numpy as np
import skimage.io as io

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
	numPts = 5
	for sample in filteredSamples:
		sample.procShape = sample.procShape[0:numPts,:]

	meanFace = pickle.load(open("meanFace.dat", "rb"))

	#Load a raw image and normalise it
	im = io.imread("00091.jpg")
	currentModel = [[3508,1445],[3592,1454],[3534,1508],[3504,1533],[3580,1539]]
	currentModel = currentModel[:numPts]
	
	tracker = pickle.load(open("tracker-5pt-halfdata.dat","rb"))
	prevFeat = None

	while 1:
		normIm = normalisedImage.NormalisedImage(im, currentModel, meanFace, {})
		normIm.CalcProcrustes()
		#print normIm.procShape
		normalisedImage.SaveNormalisedImageToFile(normIm, "img.jpg")

		print "true", normIm.procShape[:numPts]

		#model[1][0] += 0.1
		print "currentModel", currentModel

		if prevFeat is None:
			prevFeat = tracker.CalcPrevFrameFeatures(normIm, currentModel)

		normSpaceModel = [normIm.GetNormPos(*pt) for pt in currentModel]
		print "normSpaceModel", normSpaceModel

		pred = tracker.Predict(normIm, normSpaceModel, prevFeat)

		prevFeat = tracker.CalcPrevFrameFeatures(normIm, pred)

		print "pred", pred

		currentModel = [normIm.GetPixelPosImPos(*pt) for pt in pred]

