from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
import sklearn.tree.tree
import numpy as np
import simpleGbrt

if __name__=="__main__":

	boston = load_boston()

	reg = GradientBoostingRegressor()
	reg.fit(boston.data, boston.target)

	print "pred1", reg.predict([boston.data[0,:]])[0]

	featureGen = simpleGbrt.FeatureGenTest()
	featureGen.AddFeatureSet(boston.data[0,:])

	print "pred2", simpleGbrt.PredictGbrt(reg, featureGen)
	

