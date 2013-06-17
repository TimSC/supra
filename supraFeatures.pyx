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
		self.supportPixOffInitial = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))
		self.supportPixOff = self.supportPixOffInitial.copy()
		self.model = None
		self.sample = None
		self.pixGrey = None
		self.mask = None

	def Gen(self, ptNum, xOff, yOff):
		self.pixGrey = None
		self.ptNum = ptNum
		self.xOff = xOff
		self.yOff = yOff

	def SetFeatureMask(self, mask):
		self.mask = map(int, mask)
		self.supportPixOff = self.supportPixOffInitial[self.mask,:]

	def __getitem__(self, ind):
		if self.pixGrey is None:
			pix = ExtractSupportIntensity(self.sample, self.supportPixOff, \
				self.model[self.ptNum][0], self.model[self.ptNum][1], self.xOff, self.yOff)
			pix = pix.reshape((pix.shape[0],1,pix.shape[1]))
			pixGrey = col.rgb2xyz(pix)
			self.pixGrey = pixGrey.reshape(pixGrey.size)

		return self.pixGrey[ind]

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(self.supportPixOffInitial.shape[0]))

class FeatureSobel:
	def __init__(self, supportPixHalfWidth, numSupportPix=50):
		self.supportPixOffSobelInitial = np.random.uniform(low=-supportPixHalfWidth, \
			high=supportPixHalfWidth, size=(numSupportPix, 2))
		self.supportPixOffSobel = self.supportPixOffSobelInitial.copy()
		self.model = None
		self._sample = None
		self.sobelSample = None
		self.feat = None
		self.mask = None

	def Gen(self, ptNum, xOff, yOff):
		self.ptNum = ptNum
		self.xOff = xOff
		self.yOff = yOff
		self.feat = None

	def SetSample(self, sample):
		self._sample = sample
		self.sobelSample = None

	def SetFeatureMask(self, mask):
		self.mask = map(int, mask)
		self.supportPixOffSobel = self.supportPixOffSobelInitial[self.mask,:]

	def __getitem__(self, ind):
		if self.sobelSample is None:
			self.sobelSample = normalisedImageOpt.KernelFilter(self._sample)
		if self.feat is None:
			pixSobel = ExtractSupportIntensity(self.sobelSample, self.supportPixOffSobel, \
				self.model[self.ptNum][0], self.model[self.ptNum][1], self.xOff, self.yOff)
			pixConvSobel = []
			for px in pixSobel:
				pixConvSobel.extend(px)
			self.feat = pixConvSobel
		return self.feat[ind]

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(self.supportPixOffSobelInitial.shape[0]))

class FeatureHog:
	def __init__(self):
		self.model = None
		self.sample = None
		self.feat = None
		self.mask = None

	def Gen(self, ptNum, xOff, yOff):
		self.feat = None
		self.ptNum = ptNum
		self.xOff = xOff
		self.yOff = yOff

	def SetFeatureMask(self, mask):
		self.mask = map(int, mask)

	def __getitem__(self, ind):
		if self.feat is None:
			imLocs = normalisedImageOpt.GenPatchOffsetList(self.model[self.ptNum][0]+self.xOff, self.model[self.ptNum][1]+self.yOff)
			localPatch = normalisedImageOpt.ExtractPatchAtImg(self.sample, imLocs)
			localPatchGrey = col.rgb2grey(np.array([localPatch]))
			localPatchGrey = localPatchGrey.reshape((24,24)).transpose()
			self.feat = feature.hog(localPatchGrey)
		return self.feat[self.mask[ind]]

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(81))

class FeatureDists:
	def __init__(self, numPoints):
		self.model = None
		self.sample = None
		self.numPoints = numPoints
		self.modelOffset = None
		self.shapeNoise = 0.
		self.feat = None
		self.mask = None

	def Gen(self, ptNum, xOff, yOff):
		self.ptNum = ptNum
		self.xOff = xOff
		self.yOff = yOff
		self.feat = None

	def SetFeatureMask(self, mask):
		self.mask = map(int, mask)

	def __getitem__(self, ind):
		if self.feat is None:
			feat = []
			modifiedPos = np.array(self.model) + np.array(self.modelOffset)

			for i in range(len(self.modelOffset)):
				if i == self.ptNum: continue	
				dx = (modifiedPos[i,0] - modifiedPos[self.ptNum,0])
				dy = (modifiedPos[i,1] - modifiedPos[self.ptNum,1])
				if self.shapeNoise > 0.:
					dx += np.random.randn() * self.shapeNoise
					dy += np.random.randn() * self.shapeNoise
				feat.append(dx)
				feat.append(dy)
		return feat[self.mask[ind]]

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(2*(self.numPoints-1)))

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
		self.featureMap = None

	def SetImage(self, img):
		self.sample = img
		self.featureIntSupport.sample = img
		self.sobelGen.SetSample(img)
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
		self.featureIntSupport.Gen(self.ptNum, self.xOff, self.yOff)
		self.sobelGen.Gen(self.ptNum, self.xOff, self.yOff)
		self.hogGen.Gen(self.ptNum, self.xOff, self.yOff)
		self.relDistGen.Gen(self.ptNum, self.xOff, self.yOff)

		assert self.featureMask is not None
		return self.GetGenFeat()

	def GetGenFeat(self):
		out = []
		for i in range(len(self)):
			out.append(self[i])
		return out

	def __getitem__(self, ind):
		indc = self.featureMap[ind]
		module = indc[0]
		arg = indc[1]
		return module[arg]

	def __len__(self):
		return len(self.featureMap)

	def SetFeatureMask(self, mask):
		self.featureMask = mask
		compDict = {}
		self.featureMap = []

		for comp in self.featureMask:
			prefix = comp[0:3]
			cnum = comp[3:]
			module = None
			if "int" in prefix: module = self.featureIntSupport
			if "sob" in prefix: module = self.sobelGen
			if "hog" in prefix: module = self.hogGen
			if "dst" in prefix: module = self.relDistGen
			if prefix not in compDict:
				compDict[prefix] = []
			si = len(compDict[prefix])
			self.featureMap.append((module, si))
			compDict[prefix].append(cnum)
		
		for mod in ['int','sob','hog','dst']:
			if mod not in compDict:
				compDict[mod] = []

		self.featureIntSupport.SetFeatureMask(compDict['int'])
		self.sobelGen.SetFeatureMask(compDict['sob'])
		self.hogGen.SetFeatureMask(compDict['hog'])
		self.relDistGen.SetFeatureMask(compDict['dst'])

	def GetFeatureList(self):
		comp = ["int"+str(i) for i in self.featureIntSupport.GetFeatureList()]
		comp.extend(["sob"+str(i) for i in self.sobelGen.GetFeatureList()])
		comp.extend(["hog"+str(i) for i in self.hogGen.GetFeatureList()])
		comp.extend(["dst"+str(i) for i in self.relDistGen.GetFeatureList()])

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

