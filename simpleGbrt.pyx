# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport sklearn.tree._tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
#from sklearn import tree
import sklearn.tree.tree

import numpy as np
cimport numpy as np
np.import_array()

cdef class FeatureGenTest:
	
	cdef np.ndarray arr
	cdef public object generators, accessedFest
	cdef np.ndarray mapToGen, mapToInd

	def __init__(self):
		self.mapToGen = None
		self.mapToInd = None
		self.generators = []
		self.accessedFest = set()

	def __getitem__(self, key):
		return self.GetItemFast(key)

	def __len__(self):
		cdef np.ndarray[np.int32_t, ndim=1] mapToInd = self.mapToInd
		return len(mapToInd)

	cdef double GetItemFast(self, int key):
		cdef np.ndarray[np.int32_t, ndim=1] mapToInd = self.mapToInd
		cdef np.ndarray[np.int32_t, ndim=1] mapToGen = self.mapToGen
		
		if key < 0 or key >= mapToGen.shape[0]: 
			raise Exception("Invalid key "+str(key))

		self.accessedFest.add(key)

		cdef int generatorInd = mapToGen[key]
		generator = self.generators[generatorInd]
		cdef int valInd = mapToInd[key]
		cdef double val = generator[valInd]
		return val 

	def AddFeatureSet(self, generator):
		cdef np.ndarray[np.int32_t, ndim=1] mapToInd = self.mapToInd
		cdef np.ndarray[np.int32_t, ndim=1] mapToGen = self.mapToGen
	
		genNum = len(self.generators)
		self.generators.append(generator)
		l = len(generator)
		if mapToGen is None:
			mapToGen = np.array([genNum for i in range(l)], dtype=np.int32)
			mapToInd = np.array(range(l), dtype=np.int32)
			self.mapToGen = mapToGen
			self.mapToInd = mapToInd
		else:
			mapToGen = np.concatenate((mapToGen, [genNum for i in range(l)]))
			mapToInd = np.concatenate((mapToInd, range(l)))
			self.mapToGen = mapToGen
			self.mapToInd = mapToInd

	def ClearFeatureSets(self):
		self.mapToGen = None
		self.mapToInd = None
		self.generators = []		
		self.accessedFest = set()

cdef double SimplePred(FeatureGenTest features, \
	int *children_left, \
	int *children_right, \
	int *feature, \
	double *threshold, \
	double *value):

	cdef int node = 0, featNum
	cdef double featureVal

	while children_left[node] != -1 and children_right[node] != -1:	
		featNum = feature[node]
		featureVal = features.GetItemFast(featNum)
		if featureVal <= threshold[node]:
			node = children_left[node]
		else:
			node = children_right[node]
	
	return value[node]

def PredictGbrt(model, FeatureGenTest features):

	cdef float currentVal = 0., nodeVal
	cdef int i, initSet = 0
	cdef sklearn.tree._tree.Tree tree
	cdef float learn_rate = model.learn_rate

	if hasattr(model,"init"):
		currentVal = model.init.mean
		initSet = 1
	if hasattr(model,"init_"):
		currentVal = model.init_.mean
		initSet = 1
	assert initSet
	
	for i in range(model.n_estimators):
		tree = model.estimators_[i,0]
		assert tree.n_outputs == 1
		assert tree.n_classes[0] == 1

		nodeVal = SimplePred(features, \
			tree.children_left, \
			tree.children_right, \
			tree.feature, \
			tree.threshold, \
			tree.value)

		currentVal += nodeVal*learn_rate
	return currentVal

if __name__=="__main__":

	boston = load_boston()

	reg = GradientBoostingRegressor()
	reg.fit(boston.data, boston.target)

	print "pred1", reg.predict([boston.data[0,:]])[0]
	print "pred2", PredictGbrt(reg, boston.data[0,:])
	

