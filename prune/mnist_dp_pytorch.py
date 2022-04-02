# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
import pickle
from multiprocessing.pool import ThreadPool
from typing import Dict
import argparse

# # Dataset
#
# For this assignment, we will be using the popular `MNIST` dataset, which has `60000` training images and `x` testing images of digits ranging from 0-9. Each datapoing is a `28x28` greyscale image.
#
# We first load the training and testing dataset directly using PyTorch's `torchvision` library as follows:


def get_model_artifacts():
    input_size = 784
    hidden_sizes = [128, 32]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size))
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def prune(model: torch.nn.Module, mag_thres=0.01):
    """
    Performs magnitued pruning of weights
    """
    threshold = torch.nn.Threshold(threshold=mag_thres, value=0, inplace=False)
    params = model.parameters()
    with torch.no_grad():
        for param in params:
            temp = threshold(param)
            param.copy_(temp)
# %% training code

def train(model, criterion, trainset, valset, accountant, args: argparse.Namespace):

    for epoch in range(args.epochs):
        # Define accountant
        model.train()
        # for batch in DataLoader(trainset, batch_size=32):
        train_loader = DataLoader(trainset, batch_size=args.batch_size)
        # train_loader = get_dp_loader(model, optimizer, train_loader, n_epochs, t_epsilon, deltaarg, max_grad_norm)
        for batch in train_loader:
            for param in model.parameters():
                param.accumulated_grads = []

            if not args.dpsgd:
                # Perform normal weight update
                x = batch[0]
                x = x.view(x.shape[0], -1)
                y= batch[1]
                model.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y.view(-1,))
                loss.backward()  # Now p.grad for this x is filled

                # Perform weight update
                for param in model.parameters():
                    param.data.sub_(args.lr * param.grad.data)
            else:
                #Run microbatches
                for i in range(len(batch)):
                    x = batch[0][i]
                    x = x.view(x.shape[0], -1)
                    y= batch[1][i]
                    model.zero_grad()
                    y_hat = model(x)
                    loss = criterion(y_hat, y.view(-1,))
                    loss.backward()  # Now p.grad for this x is filled


                    # Clip each parameter's per-sample gradient
                    for param in model.parameters():
                        per_sample_grad = param.grad.detach().clone()
                        clip_grad_norm_(per_sample_grad, max_norm=args.max_grad_norm)  # in-place
                        param.accumulated_grads = [per_sample_grad]


                # Aggregate back
                for param in model.parameters():
                    # modified code to take the sum, otherwise the 0th dimension just keeps increasing
                    # which doesn't match the parameter dimensio
                    param.grad = torch.stack(param.accumulated_grads, dim=0).sum(0)

                for param in model.parameters():
                    param.data.sub_(args.lr * param.grad.data)

                    if args.dpsgd:
                        std = (args.noise * args.max_grad_norm) * torch.ones_like(param)
        #                 param = torch.ones_like(param) * 0.
                        param = param + torch.normal(mean=0., std=std)

    #         accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampler_rate)

        # Now do pruning
        if args.prune:
            prune_loader = DataLoader(trainset, batch_size=args.batch_size)
            prune_iter = iter(prune_loader)
            n_prune_iter = args.n_prune
            print("pruning")
            for i in range(n_prune_iter):
                batch = next(prune_iter)
                prune(model, mag_thres=args.thresh)  # parameterize this
                # Perform normal weight update
                x = batch[0]
                x = x.view(x.shape[0], -1)
                y= batch[1]
                model.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y.view(-1,))
                loss.backward()  # Now p.grad for this x is filled

                # Perform weight update
                for param in model.parameters():
                    param.data.sub_(1e-1 * param.grad.data)


        # Do one final step of pruning
        prune(model, mag_thres=args.thresh)

        # Estimate epsilon after taking a gradient step.
        accountant.step(noise_multiplier=args.noise, sample_rate=args.sampler_rate)
        epsilon = accountant.get_epsilon(delta=args.delta)

        val_acc = test(model, valset, args)
        print(f"Epoch: {epoch}, Validation acc:{val_acc.mean():.2f}, epsilon: {epsilon}")


# %% testing

def test(model, valset, args):

    model.eval()
    losses = []
    val_acc = []

    def get_val_acc(batch, batch_size=32):
        x = batch[0]; y = batch[1]
        x = x.view(x.shape[0], -1)
        with torch.no_grad():
            y_hat = model(x)


            y_label = y_hat.argmax(1)
            acc = (y_label == y).sum() / len(y_label)
            return acc

    val_loader = DataLoader(valset, batch_size=args.batch_size)

    with ThreadPool(10) as P:
        val_acc = P.map(get_val_acc, val_loader)

    # for batch in DataLoader(valset, batch_size=32):

    # val_acc.append(acc)
    val_acc = np.array(val_acc)
    return val_acc



# %% Main


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    model, criterion = get_model_artifacts()

    # Get training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-m', '--max_grad_norm', type=float, default=1.0, help='')
    parser.add_argument('-d', '--delta', default=1e-5, type=float, help='Delta for DP-SGD')
    parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('-n', '--noise', type = float, default=1.0, help='Noise multiplier for privacy')
    parser.add_argument('-s', '--sampler_rate', default=1.0, type=float, help='Sampler rate for privacy')
    parser.add_argument('-t', '--thresh', default=0.01, type=float, help='Weight pruning threshold')
    parser.add_argument('-p', '--n-prune', help='no of pruning iterations', type=int, default=200)
    parser.add_argument('--no-prune', action='store_false')
    parser.set_defaults(prune=True)
    parser.add_argument('--no-dpsgd', action='store_false')
    parser.set_defaults(dpsgd=True)
    params = parser.parse_args()
    accountant = RDPAccountant()

    train(model, criterion, trainset, valset, accountant, params)
