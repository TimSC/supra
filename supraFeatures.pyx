# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cmath, math
cimport numpy as np
import numpy as np
import skimage.color as col, skimage.feature as feature, skimage.filter as filt
import converge, normalisedImageOpt
import lazyhog
from scipy import pi, arctan2

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

	def InitFeature(self):
		pix = ExtractSupportIntensity(self.sample, self.supportPixOff, \
			self.model[self.ptNum][0], self.model[self.ptNum][1], self.xOff, self.yOff)
		pix = pix.reshape((pix.shape[0],1,pix.shape[1]))
		pixGrey = col.rgb2xyz(pix)
		self.pixGrey = pixGrey.reshape(pixGrey.size)

	def __getitem__(self, int ind):
		if self.pixGrey is None:
			self.InitFeature()

		out = self.pixGrey[ind]
		return out

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

	def InitFeature(self):
		pixSobel = ExtractSupportIntensity(self.sobelSample, self.supportPixOffSobel, \
			self.model[self.ptNum][0], self.model[self.ptNum][1], self.xOff, self.yOff)
		pixConvSobel = []
		for px in pixSobel:
			pixConvSobel.extend(px)
		self.feat = pixConvSobel

	def __getitem__(self, int ind):
		if self.sobelSample is None:
			self.sobelSample = normalisedImageOpt.KernelFilter(self._sample)
		if self.feat is None:
			self.InitFeature()

		out = self.feat[ind]
		return out

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(self.supportPixOffSobelInitial.shape[0]))

cdef class FeatureHog:
	cdef public np.ndarray model, feat, mask, cellOffsets
	cdef public object sample, featIsSet
	cdef public int ptNum
	cdef public float xOff, yOff

	def __init__(self):
		self.model = None
		self.sample = None
		self.feat = None
		self.featIsSet = False
		self.mask = None
		self.ptNum = -1
		self.xOff = 0.
		self.yOff = 0.

		#Calculate cell centre positions
		cy = 8
		cx = 8
		y = cy / 2
		x = cx / 2
		cy2 = cy * 3
		cx2 = cx * 3
		cellOffsetsLi = []

		while y < cy2:
			cellRow = []
			x = cx / 2
			while x < cx2:
				cellRow.append((x, y))
				x += cx
			y += cy
			cellOffsetsLi.append(cellRow)
		self.cellOffsets = np.array(cellOffsetsLi, dtype=np.int32)

	def Gen(self, ptNum, xOff, yOff):
		self.feat = None
		self.featIsSet = False
		self.ptNum = ptNum
		self.xOff = xOff
		self.yOff = yOff

	cdef InitFeature(self):
		cdef np.ndarray[np.float64_t, ndim=2] imLocs
		cdef int ptNum

		ptNum = self.ptNum
		imLocs = normalisedImageOpt.GenPatchOffsetList(self.model[ptNum][0]+self.xOff, self.model[ptNum][1]+self.yOff)
		localPatch = normalisedImageOpt.ExtractPatchAtImg(self.sample, imLocs)
		localPatchGrey = col.rgb2grey(np.array([localPatch]))
		localPatchGrey = localPatchGrey.reshape((24,24)).transpose()

		cdef int sy = localPatchGrey.shape[0]
		cdef int sx = localPatchGrey.shape[1]
		cdef np.ndarray[np.float64_t, ndim=2] gx = np.zeros((sy,sx))
		cdef np.ndarray[np.float64_t, ndim=2] gy = np.zeros((sy,sx))
		gx[:, :-1] = np.diff(localPatchGrey, n=1, axis=1)
		gy[:-1, :] = np.diff(localPatchGrey, n=1, axis=0)

		cdef int cx = 8
		cdef int cy = 8

		cdef np.ndarray[np.float64_t, ndim=2] magnitude = (gx**2 + gy**2) ** 0.5
		cdef np.ndarray[np.float64_t, ndim=2] orientation = arctan2(gy, gx) * (180 / 3.14159265359) % 180
		numCells = self.cellOffsets.shape[0] * self.cellOffsets.shape[1]
		magPatch = np.empty((numCells, cy, cx), dtype=np.float64)
		oriPatch = np.empty((numCells, cy, cx), dtype=np.float64)
		count = 0

		for yi in range(self.cellOffsets.shape[0]):
			for xi in range(self.cellOffsets.shape[1]):
				
				centX = self.cellOffsets[yi, xi, 0]
				centY = self.cellOffsets[yi, xi, 1]

				for y in range(cy):
					for x in range(cx):
						magPatch[count, y, x] = magnitude[y + centY - cy/2, x + centX - cx/2]
						oriPatch[count, y, x] = orientation[y + centY - cy/2, x + centX - cx/2]

				count += 1

		self.feat = lazyhog.hog(localPatchGrey, self.cellOffsets, magPatch, oriPatch)
		self.featIsSet = True

	def SetFeatureMask(self, mask):
		cdef np.ndarray[np.int32_t, ndim=1] masks = np.array(map(int, mask), dtype=np.int32)
		self.mask = masks

	def __getitem__(self, int ind):
		return self.GetItem(ind)

	cdef float GetItem(self, int ind):
		cdef np.ndarray[np.int32_t, ndim=1] masks = self.mask

		if masks is None:
			raise Exception("Masks not set")

		if self.featIsSet is False:
			self.InitFeature()

		cdef np.ndarray[np.float64_t, ndim=1] feat = self.feat
		cdef int comp = masks[ind] 
		cdef float out = feat[comp]
		return out

	def __len__(self):
		return len(self.mask)

	def GetFeatureList(self):
		return map(str,range(81))

	def __getstate__(self):
		return (self.model, self.feat, self.mask, self.sample, 
			self.featIsSet, self.ptNum, self.xOff, self.yOff, self.cellOffsets)
	
	def __setstate__(self, state):
		self.model = state[0]
		self.feat = state[1]
		self.mask = state[2]
		self.sample = state[3]
		self.featIsSet = state[4]
		self.ptNum = state[5]
		self.xOff = state[6]
		self.yOff = state[7]
		self.cellOffsets = state[8]

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
		self.mask = np.array(map(int, mask), dtype=np.int32)

	def InitFeature(self):
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

		self.feat = feat

	def __getitem__(self, ind):

		if self.feat is None:
			self.InitFeature()
			
		out = self.feat[self.mask[ind]]
		return out

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
		self.SetFeatureMask(self.GetFeatureList())

	def SetImage(self, img):
		self.sample = img
		self.featureIntSupport.sample = img
		self.sobelGen.SetSample(img)
		self.hogGen.sample = img
		self.relDistGen.sample = img

	def SetModel(self, np.ndarray[np.float64_t, ndim=2] model):
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

		#assert self.featureMask is not None
		#return self.GetGenFeat()

	def GetGenFeat(self):
		cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(len(self))
		for i in range(out.shape[0]):
			out[i] = self[i]
		return out

	def __getitem__(self, int ind):
		indc = self.featureMap[ind]
		module = indc[0]
		cdef int arg = indc[1]
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
			if module is None:
				raise Exception ("Unknown module "+prefix)
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

