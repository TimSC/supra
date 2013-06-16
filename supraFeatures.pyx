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
	
class FeatureGen:
	def __init__(self, supportPixHalfWidth, numSupportPix=50):
		self.sample = None
		self.feat = None
		self.featureIntSupport = FeatureIntSupport(supportPixHalfWidth, numSupportPix)
		self.sobelGen = FeatureSobel(supportPixHalfWidth, numSupportPix)
		self.featureMask = None

	def SetImage(self, img):
		self.sample = img
		self.featureIntSupport.sample = img
		self.sobelGen.sample = img

	def SetModel(self, model):
		self.model = model
		self.featureIntSupport.model = model
		self.sobelGen.model = model

	def SetPrevFrameFeatures(self, prevFeat):
		self.prevFrameFeatures = prevFeat

	def SetModelOffset(self, modelOffset):
		self.modelOffset = modelOffset

	def ClearModelOffset(self):
		self.modelOffset = np.zeros(np.array(self.modelOffset).shape)

	def SetShapeNoise(self, noise):
		self.shapeNoise = noise

	def SetPointNum(self, ptNum):
		self.ptNum = ptNum

	def SetOffset(self, offX, offY):
		self.xOff = offX
		self.yOff = offY

	def GenHog(self, ptNum, xOff, yOff):
		imLocs = normalisedImageOpt.GenPatchOffsetList(self.model[ptNum][0]+xOff, self.model[ptNum][1]+yOff)
		localPatch = normalisedImageOpt.ExtractPatchAtImg(self.sample, imLocs)
		localPatchGrey = col.rgb2grey(np.array([localPatch]))
		localPatchGrey = localPatchGrey.reshape((24,24)).transpose()
		return feature.hog(localPatchGrey)

	def GenDistancePairs(self, ptNum, xOff, yOff):
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

	def Gen(self):

		self.pixGreyNorm = self.featureIntSupport.Gen(self.ptNum, self.xOff, self.yOff)
		self.pixNormSobel = self.sobelGen.Gen(self.ptNum, self.xOff, self.yOff)
		hog = self.GenHog(self.ptNum, self.xOff, self.yOff)
		relDist = self.GenDistancePairs(self.ptNum, self.xOff, self.yOff)

		self.feat = np.concatenate([hog, self.prevFrameFeatures, relDist])
		print self.feat.shape

		if self.featureMask is None:
			return self.feat

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
			if prefix == "aaa":
				return self.feat[cnum]
			if prefix == "int":
				return self.pixGreyNorm[cnum]
			if prefix == "sob":
				return self.pixNormSobel[cnum]

		else:
			return self.feat[ind]

	def __len__(self):
		if self.featureMask is not None:
			return len(self.featureMask)
		return len(self.feat)

	def SetFeatureMask(self, mask):
		self.featureMask = mask

	def GetFeatureList(self):
		comp = ["aaa"+str(i) for i in range(114)]
		comp.extend(self.featureIntSupport.GetFeatureList())
		comp.extend(self.sobelGen.GetFeatureList())
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

