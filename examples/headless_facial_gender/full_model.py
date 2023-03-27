import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        # Model used for training in Pytorch only. 
        # No softmax because it is done in the loss function
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.Batch1=nn.BatchNorm2d(16)
        self.Batch2=nn.BatchNorm2d(32)
        self.Batch3=nn.BatchNorm2d(64)
        self.Batch4=nn.BatchNorm2d(128)
    
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(128 * 1 * 1, 128) 
        self.fc2=nn.Linear(128,256)
        self.fc3=nn.Linear(256,2)

    def forward(self, x):
 
        out = self.cnn1(x) 
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch1(out)
 
        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch2(out)
        
        out = self.cnn3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch3(out)
        
        out = self.cnn4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Batch4(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        out = self.fc2(out)
        
        out = self.fc3(out)
        
        return out

