import numpy as np
import skimage.color as col, skimage.feature as feature, skimage.filter as filt
import converge, normalisedImageOpt

def ExtractSupportIntensity(normImage, supportPixOff, ptX, ptY, offX, offY):
	supportPixOff = supportPixOff.copy()
	supportPixOff += [offX, offY]
	supportPixOff += [ptX, ptY]
	return normImage.GetPixelsImPos(supportPixOff)

############# Feature Generation #####################

class FeatureIntSupport:
	def __init__(self, supportPixHalfWidth, numSupportPix=50):
		self.supportPixOff = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))
		self.model = None
		self.sample = None
		self.numSupportPix = numSupportPix

	def Gen(self, ptNum, xOff, yOff):
		pix = ExtractSupportIntensity(self.sample, self.supportPixOff, \
			self.model[ptNum][0], self.model[ptNum][1], xOff, yOff)
		pix = pix.reshape((pix.shape[0],1,pix.shape[1]))
		pixGrey = col.rgb2xyz(pix)
		pixGrey = pixGrey.reshape(pixGrey.size)
		
		return pixGrey

	def GetFeatureList(self):
		return ["int"+str(num) for num in range(self.numSupportPix)]

class FeatureSobel:
	def __init__(self, supportPixHalfWidth, numSupportPix=50):
		self.supportPixOffSobel = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))
		self.model = None
		self.sample = None
		self.numSupportPix = numSupportPix
		self.sobelSample = None

	def Gen(self, ptNum, xOff, yOff):
		if self.sobelSample is None:
			self.sobelSample = normalisedImageOpt.KernelFilter(self.sample)
		pixSobel = ExtractSupportIntensity(self.sobelSample, self.supportPixOffSobel, \
			self.model[ptNum][0], self.model[ptNum][1], xOff, yOff)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)
		return pixConvSobel

	def GetFeatureList(self):
		return ["sob"+str(num) for num in range(self.numSupportPix)]

class FeatureHog:
	def __init__(self):
		self.model = None
		self.sample = None

	def Gen(self, ptNum, xOff, yOff):
		imLocs = normalisedImageOpt.GenPatchOffsetList(self.model[ptNum][0]+xOff, self.model[ptNum][1]+yOff)
		localPatch = normalisedImageOpt.ExtractPatchAtImg(self.sample, imLocs)
		localPatchGrey = col.rgb2grey(np.array([localPatch]))
		localPatchGrey = localPatchGrey.reshape((24,24)).transpose()
		feat = feature.hog(localPatchGrey)
		return feat

	def GetFeatureList(self):
		return ["sob"+str(num) for num in range(81)]

class FeatureDists:
	def __init__(self, numPoints):
		self.model = None
		self.sample = None
		self.numPoints = numPoints
		self.modelOffset = None
		self.shapeNoise = 0.

	def Gen(self, ptNum, xOff, yOff):
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

	def GetFeatureList(self):
		return ["dst"+str(i) for i in range(2*(self.numPoints-1))]

class FeatureGen:
	def __init__(self, numPoints, supportPixHalfWidth, numSupportPix=50):
		self.sample = None
		self.feat = None
		self.model = None
		self.featureIntSupport = FeatureIntSupport(supportPixHalfWidth, numSupportPix)
		self.sobelGen = FeatureSobel(supportPixHalfWidth, numSupportPix)
		self.hogGen = FeatureHog()
		self.relDistGen = FeatureDists(numPoints)
		self.featureMask = None
		self.numPoints = numPoints

	def SetImage(self, img):
		self.sample = img
		self.featureIntSupport.sample = img
		self.sobelGen.sample = img
		self.hogGen.sample = img
		self.relDistGen.sample = img

	def SetModel(self, model):
		self.model = model
		self.featureIntSupport.model = model
		self.sobelGen.model = model
		self.hogGen.model = model
		self.relDistGen.model = model

	def SetModelOffset(self, modelOffset):
		self.modelOffset = modelOffset
		self.relDistGen.modelOffset = modelOffset

	def ClearModelOffset(self):
		self.modelOffset = np.zeros(np.array(self.modelOffset).shape)
		self.relDistGen.modelOffset = self.modelOffset

	def SetShapeNoise(self, noise):
		self.relDistGen.shapeNoise = noise

	def SetPointNum(self, ptNum):
		self.ptNum = ptNum

	def SetOffset(self, offX, offY):
		self.xOff = offX
		self.yOff = offY

	def Gen(self):
		self.pixGreyNorm = self.featureIntSupport.Gen(self.ptNum, self.xOff, self.yOff)
		self.pixNormSobel = self.sobelGen.Gen(self.ptNum, self.xOff, self.yOff)
		self.hog = self.hogGen.Gen(self.ptNum, self.xOff, self.yOff)
		self.relDist = self.relDistGen.Gen(self.ptNum, self.xOff, self.yOff)

		assert self.featureMask is not None
		return self.GetGenFeat()

	def GetGenFeat(self):
		out = []
		for i in range(len(self)):
			out.append(self[i])
		return out

	def __getitem__(self, ind):
		if 	self.featureMask is not None:
			indc = self.featureMask[ind]
			prefix = indc[0:3]
			cnum = int(indc[3:])
			if prefix == "int":
				return self.pixGreyNorm[cnum]
			if prefix == "sob":
				return self.pixNormSobel[cnum]
			if prefix == "hog":
				return self.hog[cnum]
			if prefix == "dst":
				return self.relDist[cnum]
		else:
			return self.feat[ind]

	def __len__(self):
		return len(self.featureMask)

	def SetFeatureMask(self, mask):
		self.featureMask = mask

	def GetFeatureList(self):
		comp = self.featureIntSupport.GetFeatureList()
		comp.extend(self.sobelGen.GetFeatureList())
		comp.extend(self.hogGen.GetFeatureList())
		comp.extend(self.relDistGen.GetFeatureList())
		return comp

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

