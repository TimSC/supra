# cython: profile=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn import tree

cdef SimplePred(tree, features):
	assert tree.n_classes == 1
	assert tree.n_outputs == 1
	cdef int node = 0

	while tree.children_left[node] != -1 and tree.children_right[node] != -1:	
		featNum = tree.feature[node]
		if features[featNum] <= tree.threshold[node]:
			node = tree.children_left[node]
		else:
			node = tree.children_right[node]
	
	return tree.value[node][0][0]

def PredictGbrt(model, features):

	currentVal = None
	if hasattr(model,"init"):
		currentVal = model.init.mean
	if hasattr(model,"init_"):
		currentVal = model.init_.mean
	assert currentVal is not None
	
	for i in range(model.n_estimators):
		#test = SimplePred(model.estimators_[i,0].tree_, features)
		test = SimplePred(model.estimators_[i,0], features)
		#currentVal += test*model.learning_rate
		currentVal += test*model.learn_rate
	return currentVal

if __name__=="__main__":

	boston = load_boston()

	reg = GradientBoostingRegressor()
	reg.fit(boston.data, boston.target)

	print "pred1", reg.predict([boston.data[0,:]])[0]
	print "pred2", PredictGbrt(reg, boston.data[0,:])
	

