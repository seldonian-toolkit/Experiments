# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
# from facial_recog_cnn import PytorchFacialRecog
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)

import torch

if __name__ == "__main__":
	torch.manual_seed(0)
	regime='supervised_learning'
	sub_regime='classification'
	
	N=23700 # Clips off 5 samples (at random) to make total divisible by 150,
	# the desired batch size
	
	# Check for raw data
	raw_file = "./data/raw/age_gender.csv"
	if not os.path.exists(raw_file):
		raise FileNotFoundError(
			"Raw file './data/raw/age_gender.csv' was not found.\n"
			"Download it from here: https://github.com/seldonian-toolkit/Tutorials/raw/main/tutorial_j_materials/age_gender.zip \n"
			"Then unzip it and put the csv file 'age_gender.csv' into this directory: examples/headless_facial_gender/data/raw/ (create the directory if necessary)")

	# Get the data, load from file if already saved
	savename_features = './data/proc/features.pkl'
	savename_labels = './data/proc/labels.pkl'
	savename_sensitive_attrs = './data/proc/sensitive_attrs.pkl'
	os.makedirs('./data/proc',exist_ok=True)
	if not all([os.path.exists(x) for x in [savename_features,
			savename_labels,savename_sensitive_attrs]]):
		print("Loading data...")
		data = pd.read_csv(raw_file)
		print("Done.")
		# Shuffle data since it is in order of age, then gender
		data = data.sample(n=len(data),random_state=42).iloc[:N]
		print("Randomly sampled data, first 20 points")
		print(data.iloc[0:20])
		# Convert pixels from string to numpy array
		print("Converting pixels to array...")
		data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
		print("Done.")
		# normalize pixels data
		print("Normalizing and reshaping pixel data...")
		data['pixels'] = data['pixels'].apply(lambda x: x/255)
		print("Done.")
		# Reshape pixels array
		X = np.array(data['pixels'].tolist())
		# Converting pixels from 1D to 4D
		features = X.reshape(X.shape[0],1,48,48)
		labels = data['gender'].values
		
		M=data['gender'].values
		mask=~(M.astype("bool"))
		F=mask.astype('int64')
		sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))
		print("Saving features, labels, and sensitive_attrs to pickle files")
		assert len(features) == N
		assert len(labels) == N
		assert len(sensitive_attrs) == N
		save_pickle(savename_features,features,verbose=True)
		save_pickle(savename_labels,labels,verbose=True)
		save_pickle(savename_sensitive_attrs,sensitive_attrs,verbose=True)
		print("Done.")
	
	
	