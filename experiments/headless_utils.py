""" Utilities used in the rest of the library """

import os
import pickle
import numpy as np
import math

from seldonian.RL.RL_runner import create_agent, run_trial_given_agent_and_env
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.dataset import SupervisedDataSet

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch import optim
from torch.autograd import Variable

from seldonian.utils.io_utils import load_pickle, save_pickle


def train_pytorch_model(pytorch_model, num_epochs, data_loaders, optimizer, loss_func):
    """Train a pytorch model

    :param pytorch_model: The PyTorch model object. Must have a .forward() method

    :param num_epochs: Number of epochs to train for
    :type num_epochs: int

    :param data_loaders: Dictionary containing data loaders, with keys:
            'candidate' and 'safety', each pointing to torch dataloaders.
            Each data loader should only have features and labels in them.

    :param optimizer: The PyTorch optimizer to use

    :param loss_func: The PyTorch loss function
    """
    pytorch_model.train()

    # Train the pytorch_model
    total_step = len(data_loaders["candidate"])

    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(data_loaders["candidate"]):
            features = features.to(device)
            labels = labels.to(device)
            b_x = Variable(features)  # batch x
            output = pytorch_model(b_x)
            b_y = Variable(labels)  # batch y
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )


def make_data_loaders(
    features, labels, frac_data_in_safety, candidate_batch_size, safety_batch_size
):
    """
    Create PyTorch data loaders for candidate and safety datasets
    """
    n_points_tot = len(features)
    n_candidate = int(round(n_points_tot * (1.0 - frac_data_in_safety)))
    n_safety = n_points_tot - n_candidate

    F_c = features[:n_candidate]
    F_s = features[n_candidate:]
    # Split labels - must be numpy array
    L_c = labels[:n_candidate]
    L_s = labels[n_candidate:]

    F_c_tensor = torch.from_numpy(F_c)
    F_s_tensor = torch.from_numpy(F_s)
    L_c_tensor = torch.from_numpy(L_c)
    L_s_tensor = torch.from_numpy(L_s)

    dataset_c = torch.utils.data.TensorDataset(F_c_tensor, L_c_tensor)

    dataloader_c = torch.utils.data.DataLoader(
        dataset_c, batch_size=candidate_batch_size, shuffle=False
    )

    dataset_s = torch.utils.data.TensorDataset(F_s_tensor, L_s_tensor)

    dataloader_s = torch.utils.data.DataLoader(
        dataset_s, batch_size=safety_batch_size, shuffle=False
    )

    data_loaders = {"candidate": dataloader_c, "safety": dataloader_s}
    return data_loaders

def update_headless_state_dict(
    full_state_dict,
    headless_model,
    head_layer_names):
    """
    
    """
    # Remove entries in full_state_dict
    # that are from the head of the model
    for layer_name in head_layer_names:
        del full_state_dict[layer_name]
    # Use this to update state dict of headless model 
    headless_model.load_state_dict(full_state_dict)

def create_latent_features(
    loaders,
    headless_model,
    latent_feature_shape,
    ):
    """

    """
    n_points_tot = len(loaders['candidate'].dataset) + len(loaders['safety'].dataset)
    candidate_batch_size = loaders['candidate'].batch_size
    safety_batch_size = loaders['safety'].batch_size
    
    new_features_shape = (n_points_tot, *(latent_feature_shape))
    new_features = np.zeros(new_features_shape)

    # Pass candidate data through first
    end_index = 0
    for batch_features, labels in loaders["candidate"]:
        start_index = end_index
        end_index = start_index + len(batch_features)
        batch_features = batch_features.to(device)
        new_features[start_index:end_index] = (
            headless_model(batch_features).cpu().detach().numpy()
        )
    # Now pass safety data through
    for batch_features, labels in loaders["safety"]:
        start_index = end_index
        end_index = start_index + len(batch_features)
        batch_features = batch_features.to(device)
        new_features[start_index:end_index] = (
            headless_model(batch_features).cpu().detach().numpy()
        )

    return new_features 

def generate_latent_features(
    full_model,
    headless_model,
    head_layer_names,
    orig_features,
    labels, 
    frac_data_in_safety, 
    candidate_batch_size, 
    safety_batch_size,
    loss_func=nn.CrossEntropyLoss(),
    learning_rate=0.001,
    num_epochs=5,
    device=torch.device("cpu")):
    """

    """
    
    full_model.to(device)
    headless_model.to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)

    # Make data loaders
    data_loaders = make_data_loaders(
        features=orig_features,
        labels=labels, 
        frac_data_in_safety=frac_data_in_safety, 
        candidate_batch_size=candidate_batch_size, 
        safety_batch_size=safety_batch_size
    )

    # Train model
    train_pytorch_model(
        pytorch_model=full_model,
        num_epochs=num_epochs, 
        data_loaders=data_loaders, 
        optimizer=optimizer, 
        loss_func=loss_func)

    # get state dict after training 
    sd_after_training = cnn.state_dict()

    # Copy over only headless layers from full model's
    # state dictionary to the headless model
    update_headless_state_dict(
        full_state_dict,
        headless_model,
        head_layer_names)
    

    # Pass all candidate and safety data through the headless model
    # to create latent features

    latent_features = create_latent_features(
        data_loaders,
        headless_model,
        latent_feature_shape,
    )

    return latent_features
    

