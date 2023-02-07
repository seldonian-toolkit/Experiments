import autograd.numpy as np
from seldonian.models.models import ClassificationModel

class WeightedRandomClassifierModel(ClassificationModel):
	def __init__(self,weight):
		""" Implements a classifier that always predicts
		that the positive class has prob=0.5,
		regardless of input """
		super().__init__()
		self.has_intercept = False
		assert 0.0 <= weight <= 1.0
		self.weight = weight

	def predict(self,theta,X):
		""" Predict the probability of 
		having the positive class label

		:param theta: The parameter weights
		:type theta: numpy ndarray
		:param X: The features
		:type X: numpy ndarray
		:return: predictions for each observation
		:rtype: float
		"""
		return self.weight*np.ones(len(X))