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
	cdef float val

	def __init__(self, arrIn):
		self.arr = np.array(arrIn, dtype=np.float32)

	def __getitem__(self, key):
		cdef np.ndarray[np.float32_t, ndim=1] arr = self.arr
		return arr[key]

	def __len__(self):
		cdef np.ndarray[np.float32_t, ndim=1] arr = self.arr
		return len(arr)

	cdef float GetItemFast(self, int key):
		cdef np.ndarray[np.float32_t, ndim=1] arr = self.arr
		val = arr[key]
		return val

cdef double SimplePred(FeatureGenTest features, \
	int *children_left, \
	int *children_right, \
	int *feature, \
	double *threshold, \
	double *value):

	cdef int node = 0, featNum
	cdef float featureVal

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
	

