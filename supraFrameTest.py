
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
	for sample in filteredSamples:
		sample.procShape = sample.procShape[0:1,:]

	meanFace = pickle.load(open("meanFace.dat", "rb"))

	#Load a raw image and normalise it
	im = io.imread("00001.jpg")
	
	normIm = normalisedImage.NormalisedImage(im, [[90,203],[162,198],[138,237],[94,265],[174,259]], meanFace, {})
	normIm.CalcProcrustes()
	print normIm.procShape
	normalisedImage.SaveNormalisedImageToFile(normIm, "img.jpg")

	tracker = pickle.load(open("tracker-1pt-save.dat","rb"))

	sample = filteredSamples[0]
	model = sample.procShape
	print "true", model

	model[0][0] += 0.1
	print "perturbed", model

	prevFeat = tracker.CalcPrevFrameFeatures(sample, model)

	pred = tracker.Predict(sample, model, prevFeat)
	print "pred", pred



