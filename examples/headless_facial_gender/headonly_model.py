from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn

class FacialRecogHeadCNNModel(nn.Module):
	def __init__(self):
		super(FacialRecogHeadCNNModel, self).__init__()

		self.fc3 = nn.Linear(256,2)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		
		out = self.fc3(x)
		out = self.softmax(out)[:,1] 

		return out

class CNNHead(SupervisedPytorchBaseModel):
	def __init__(self,device):
		""" Implements just the head of the CNN, i.e.
		a single linear + softmax output layer 

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		super().__init__(device)

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		"""
		return FacialRecogHeadCNNModel()