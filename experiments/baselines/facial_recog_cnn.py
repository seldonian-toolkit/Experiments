import torch.nn as nn
import torch
from scipy.special import softmax

from experiments.experiment_utils import batch_predictions

from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from .baselines import SupervisedExperimentBaseline


class FacialRecogCNNModel(nn.Module):
    def __init__(self):
        super(FacialRecogCNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Max pool 1
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.Batch1 = nn.BatchNorm2d(16)
        self.Batch2 = nn.BatchNorm2d(32)
        self.Batch3 = nn.BatchNorm2d(64)
        self.Batch4 = nn.BatchNorm2d(128)

        self.Drop1 = nn.Dropout(0.01)
        self.Drop2 = nn.Dropout(0.5)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(128 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch1(out)
        # out=self.Drop1(out)
        # print("out:")
        # print(out)
        # input("next")

        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch2(out)
        # out=self.Drop1(out)

        out = self.cnn3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch3(out)
        # out=self.Drop1(out)

        out = self.cnn4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch4(out)
        # out=self.Drop1(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, start_dim=1)

        # Linear function (readout)
        out = self.fc1(out)

        # out=self.Drop2(out)

        out = self.fc2(out)

        # out=self.Drop2(out)

        out = self.fc3(out)
        # out=self.softmax(out)[:,1]

        return out


class PytorchFacialRecogBaseline(
    SupervisedPytorchBaseModel, SupervisedExperimentBaseline
):
    def __init__(self, device, learning_rate, batch_epoch_dict={}):
        """Implements a CNN with PyTorch.
        CNN consists of four hidden layers followed
        by a linear + softmax output layer

        """
        SupervisedPytorchBaseModel.__init__(self, device)
        SupervisedExperimentBaseline.__init__(self, model_name="facial_recog_cnn")
        self.eval_batch_size = (
            2000  # How many samples are evaluated in a single (batched) forward pass
        )
        self.batch_epoch_dict = batch_epoch_dict
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.pytorch_model.parameters(), lr=learning_rate
        )

    def create_model(self, **kwargs):
        """Create the pytorch model and return it
        Inputs are N,1,28,28 where N is the number of them,
        1 channel and 28x28 pixels.
        Do Conv2d,ReLU,maxpool twice then
        output in a fully connected layer to 10 output classes
        """
        return FacialRecogCNNModel()

    def predict(self, solution, X, **kwargs):
        y_pred_super = super().predict(solution, X, **kwargs)
        y_pred = softmax(y_pred_super, axis=-1)[:, 1]
        return y_pred

    def train(self, X_train, Y_train, batch_size, n_epochs):
        loss_list = []
        accuracy_list = []
        iter_list = []
        x_train_tensor = torch.from_numpy(X_train)
        y_train_label = torch.from_numpy(Y_train)
        train = torch.utils.data.TensorDataset(x_train_tensor, y_train_label)
        trainloader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True
        )
        itot = 0
        for epoch in range(n_epochs):
            for i, (images, labels) in enumerate(trainloader):
                # Load images
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = images.requires_grad_()

                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = self.pytorch_model(images)

                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                self.optimizer.step()
                if i % 100 == 0:
                    it = f"{i+1}/{len(trainloader)}"
                    print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{loss}")
                itot += 1
        solution = self.get_model_params()
        return solution
