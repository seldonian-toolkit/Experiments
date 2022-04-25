import numpy as np

def precalc(dataset):
	d = {}
	d['M_mask'] = dataset.df['M']==1
	return d

def demographic_parity_mean(features,labels,prediction,**kwargs):	
	precalc_dict = kwargs['precalc_dict']
	M_mask = precalc_dict['M_mask']
	prediction_M = prediction[M_mask]
	prediction_F = prediction[~M_mask]
	PR_M = np.mean(prediction_M==1.0)
	PR_F = np.mean(prediction_F==1.0)
	return abs(PR_M-PR_F) - 0.15 

constraints = [demographic_parity_mean]