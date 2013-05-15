
import urllib2
import json, pickle, math
import numpy as np
import matplotlib.pyplot as plt
import procrustes

if __name__ == "__main__":

	urlHandle = urllib2.urlopen("http://192.168.1.2/photodb/getsamples.php")
	sampleJson = urlHandle.read()

	sampleList = json.loads(sampleJson)

	scaledSamples = []
	for sample in sampleList:
		if sample['model'] is None: continue
		modelArr = np.array(sample['model'])
		centre = modelArr.mean(axis=0)
		modelCent = modelArr - centre
		axisVar = modelCent.var(axis=0)
		axisVarCombined = axisVar.mean()
		modelScaled = modelCent / (axisVarCombined ** 0.5)

		scaledSamples.append(modelScaled)

	scaledSamplesArr = np.array(scaledSamples)
	meanFace = scaledSamplesArr.mean(axis=0)
	print meanFace

	pickle.dump(meanFace, open("meanFace.dat", "wb"), protocol=-1)

	#plt.plot(meanFace[:,0],-meanFace[:,1],'x')
	#plt.show()

	if 0:
		for sample in sampleList:
			if sample['model'] is None: continue
			modelArr = np.array(sample['model'])
			procShape, params = procrustes.CalcProcrustesOnFrame(procrustes.FrameToArray(modelArr), procrustes.FrameToArray(meanFace))

			procSpaceModel = procrustes.ToProcSpace(modelArr, params)
			imgSpaceModel = procrustes.ToImageSpace(procSpaceModel, params)

