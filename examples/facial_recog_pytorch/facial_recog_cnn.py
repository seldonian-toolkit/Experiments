from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
import torch.nn as nn
import torch

class FacialRecogCNNModel(nn.Module):
	def __init__(self):
		super(FacialRecogCNNModel, self).__init__()

		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
		self.relu = nn.ReLU()

		# Max pool 1
		self.maxpool = nn.MaxPool2d(kernel_size=2)
		
		self.Batch1=nn.BatchNorm2d(16)
		self.Batch2=nn.BatchNorm2d(32)
		self.Batch3=nn.BatchNorm2d(64)
		self.Batch4=nn.BatchNorm2d(128)
		
		self.Drop1=nn.Dropout(0.01)
		self.Drop2=nn.Dropout(0.5)


		self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
		self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
		self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		
		# Fully connected 1 (readout)
		self.fc1 = nn.Linear(128 * 1 * 1, 128) 
		self.fc2=nn.Linear(128,256)
		self.fc3=nn.Linear(256,2)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		out = self.cnn1(x) 
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch1(out)
		# out=self.Drop1(out)
		# print("out:")
		# print(out)
		# input("next")

		
		out = self.cnn2(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch2(out)
		# out=self.Drop1(out)
		
		out = self.cnn3(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch3(out)
		# out=self.Drop1(out)
		
		out = self.cnn4(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out=self.Batch4(out)
		# out=self.Drop1(out)
		
		# Resize
		# Original size: (100, 32, 7, 7)
		# out.size(0): 100
		# New out size: (100, 32*7*7)
		# out = out.view(out.size(0), -1)
		out = torch.flatten(out,start_dim=1)


		# Linear function (readout)
		out = self.fc1(out)
		
		# out=self.Drop2(out)
		
		out=self.fc2(out)
		
		# out=self.Drop2(out)
		
		out=self.fc3(out)
		out=self.softmax(out)[:,1] 

		return out

class PytorchFacialRecog(SupervisedPytorchBaseModel):
	def __init__(self,device):
		""" Implements a CNN with PyTorch. 
		CNN consists of two hidden layers followed 
		by a linear + softmax output layer 

		:param input_dim: Number of features
		:param output_dim: Size of output layer (number of label columns)
		"""
		learning_rate = 0.01
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.pytorch_model.parameters(),
			lr=learning_rate)
		super().__init__(device)

	def create_model(self,**kwargs):
		""" Create the pytorch model and return it
		Inputs are N,1,28,28 where N is the number of them,
		1 channel and 28x28 pixels.
		Do Conv2d,ReLU,maxpool twice then
		output in a fully connected layer to 10 output classes
		"""
		return FacialRecogCNNModel()

	def train(self,X_train,Y_train,batch_size,num_epochs):
		print("Training model...")
		loss_list=[]
		accuracy_list=[]
		iter_list=[]
		x_train_tensor=torch.from_numpy(X_train)
		print("x_train tensor size:",x_train_tensor.size())
		y_train_label=torch.from_numpy(y_train)
		print("y_train_label size:",y_train_label.size())
		train=torch.utils.data.TensorDataset(x_train_tensor,y_train_label) 
		trainloader=torch.utils.data.DataLoader(train,
			batch_size=batch_size,shuffle=True) 
		iter = 0
		for epoch in range(num_epochs):
			for i, (images, labels) in enumerate(trainloader):
				# Load images
				images = images.requires_grad_()
				images.to(self.device)
				labels.to(self.device)

				# Clear gradients w.r.t. parameters
				optimizer.zero_grad()

				# Forward pass to get output/logits
				outputs = model(images.float())

				# Calculate Loss: softmax --> cross entropy loss
				loss = self.criterion(outputs, labels)

				# Getting gradients w.r.t. parameters
				loss.backward()

				# Updating parameters
				self.optimizer.step()

				iter += 1

				if iter % 10 == 0:
					# Calculate Accuracy         
					correct = 0
					total = 0
					# Iterate through test dataset
					for images, labels in testloader:
						# Load images
						images = images.requires_grad_()

						# Forward pass only to get logits/output
						outputs = model(images.float())

						# Get predictions from the maximum value
						_, predicted = torch.max(outputs.data, 1)

						# Total number of labels
						total += labels.size(0)

						# Total correct predictions
						correct += (predicted == labels).sum()

					accuracy = 100 * correct // total

					# Print Loss
					accuracy_list.append(accuracy)
					loss_list.append(loss.item())
					iter_list.append(iter)
					print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))