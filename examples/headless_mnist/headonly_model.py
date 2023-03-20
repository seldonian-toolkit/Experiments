from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn

class CNNHead(SupervisedPytorchBaseModel):
	def __init__(self,device):
		""" Implements the head of the full CNN network 
		which is just a linear + softmax output layer 

		"""
		super().__init__(device)

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		Inputs are N,1568 where N is the batch size.
		"""
		cnn = nn.Sequential(         
			nn.Linear(32 * 7 * 7, 10),
			nn.Softmax(dim=1)
		)       
		return cnn