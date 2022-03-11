import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import haiku
import optax
import jax
import jax.numpy as jnp
import numpy as np
from opacus.accountants.rdp import RDPAccountant
import matplotlib.pyplot as plt
import pandas as pd
from arguments import get_arg_parser
from model.fixup_resnet import ResNet9

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


def net_fn(x, is_training=True):
    net = ResNet9(num_classes=10)
    return net(x, is_training)


@jax.jit
def loss_fn(params, x, y, is_training=True):
    logits = model.apply(params, x, is_training=is_training)
    # return optax.softmax_cross_entropy(logits, y).mean()
    labels = jax.nn.one_hot(y, 10)
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    return softmax_xent


@jax.jit
def update(params, grads, opt_state):
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


@jax.jit
def predictions(params, x, y):
    pred_y = model.apply(params, x, is_training=False)
    correct = jnp.sum(jnp.argmax(pred_y, axis=-1) == y)
    return correct, x.shape[0]


@jax.jit
def get_loss_grads(params, x, y):
    x = jnp.expand_dims(x, axis=1)
    y = jnp.expand_dims(y, axis=1)
    get_value_grads = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0, 0))
    loss, grads = get_value_grads(params, x, y)
    return loss, grads


@jax.jit
def clip_grads(grads, max_clipping_value):
    grads_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
        grads, 0
    ))
    grads = jax.tree_util.tree_map(
        lambda x: x / jnp.maximum(
            1,
            jnp.expand_dims(grads_norm, axis=tuple(range(1, len(x.shape)))) / max_clipping_value
        ),
        grads
    )
    return grads


@jax.jit
def noise_grads(grads, max_clipping_value, noise_multiplier, lot_size):
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    (*rngs,) = jax.random.split(next(prng_seq), len(grads_flat))
    grads = [
        g + (max_clipping_value * noise_multiplier) * jax.random.normal(r, g.shape) \
        for r, g in zip(rngs, grads_flat)
    ]
    grads = [g / lot_size for g in grads]
    return jax.tree_unflatten(grads_treedef, grads)


if __name__ == '__main__':
    argParser = get_arg_parser()
    args = argParser.parse_args()

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    train_dataset = CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
    )

    test_dataset = CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = haiku.without_apply_rng(haiku.transform(net_fn))

    # Inits
    prng_seq = haiku.PRNGSequence(args.seed)
    optimizer = optax.sgd(learning_rate=args.lr, momentum=args.momentum)
    dummy_input = next(iter(train_loader))[0]
    dummy_input = dummy_input.cpu().detach().numpy()
    dummy_input = jnp.asarray(dummy_input)
    params = model.init(next(prng_seq), dummy_input)
    opt_state = optimizer.init(params)
    privacy_accountant = RDPAccountant()
    N_train = len(train_loader.dataset)

    Accuracy = []
    Epsilon = []
    Test_accuracy = []
    Test_epsilon = []

    for e in range(args.epochs):
        gradients = None
        correct_preds = 0
        total_preds = 0
        for i, batch in enumerate(train_loader):
            # Processing data format for jax
            batch_x, batch_y = batch
            batch_x = batch_x.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()
            batch_x = jnp.asarray(batch_x)
            batch_y = jnp.asarray(batch_y)

            # Prediction
            correct, total = predictions(params, batch_x, batch_y)
            correct_preds += correct
            total_preds += total

            # Compute gradient (forward + backward)
            loss, grads = get_loss_grads(params, batch_x, batch_y)

            # Clip (& Accumulate) gradient
            grads = clip_grads(grads, max_clipping_value=args.clip)
            if gradients is None:
                grads = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), grads)
            else:
                grads = jax.tree_util.tree_multimap(
                    lambda x, y: x + jnp.sum(y, axis=0),
                    gradients, grads
                )
            gradients = grads

            if (i + 1) % (args.lot_size // args.batch_size) == 0:
                # Add noise
                grads = noise_grads(
                    grads,
                    max_clipping_value=args.clip,
                    noise_multiplier=args.noise_multiplier,
                    lot_size=args.lot_size
                )
                # Descent (update)
                params, opt_state = update(params, grads, opt_state)
                # Privacy Accountant
                privacy_accountant.step(noise_multiplier=args.noise_multiplier,
                                        sample_rate=args.lot_size / N_train)
                eps = privacy_accountant.get_epsilon(delta=args.delta)
                # Logging
                print('Epoch: {}, Batch: {}, Acc = {:.3f}, Eps = {:.3f}'.format(
                    e, i, correct_preds / total_preds, eps
                ))
                Accuracy.append(correct_preds / total_preds)
                Epsilon.append(eps)
                # Reset
                gradients = None
                correct_preds = 0
                total_preds = 0

        # evaluate test accuracy
        correct_preds = 0
        total_preds = 0
        for test_i, test_batch in enumerate(test_loader):
            # Processing data format for jax
            test_batch_x, test_batch_y = test_batch
            test_batch_x = test_batch_x.cpu().detach().numpy()
            test_batch_y = test_batch_y.cpu().detach().numpy()
            test_batch_x = jnp.asarray(test_batch_x)
            test_batch_y = jnp.asarray(test_batch_y)
            # test pred
            correct, total = predictions(params, test_batch_x, test_batch_y)
            correct_preds += correct
            total_preds += total
        print('Test epoch {}, Test Acc = {:.3f}, Cur_eps = {:.3f}'.format(
            e, correct_preds / total_preds, eps
        ))
        Test_accuracy.append(correct_preds / total_preds)
        Test_epsilon.append(eps)
