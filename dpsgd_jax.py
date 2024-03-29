import time
import random
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import haiku
import optax
import jax
import jax.numpy as jnp
import numpy as np
from opacus.accountants.rdp import RDPAccountant
from arguments import get_arg_parser
from model.fixup_resnet import ResNet9, ResNet101
from model.cnn import CNNSmall, CNNMed, VGG16
from prune.prune import L1Unstructured
from haiku._src.data_structures import FlatMap

from util.dataloader import Normalize, AddChannelDim, Cast, NumpyLoader
from util.dp_utils import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


def net_fn(x, is_training=True):
    if args.model == 'resnet9':
        net = ResNet9(num_classes=10)
        # net = ResNet101(num_classes=10)
    elif args.model == 'cnn_small':
        classifier_config = {'activation': 'tanh', 'num_classes': 10, 'dropout_rate': None}
        net = CNNSmall(classifier_config)
    elif args.model == 'cnn_med':
        classifier_config = {'activation': 'tanh', 'num_classes': 10, 'dropout_rate': None}
        net = CNNMed(classifier_config)
    elif args.model == 'vgg16':
        classifier_config = {'activation': 'tanh', 'num_classes': 10, 'dropout_rate': 0.5}
        # classifier_config = {'activation': 'tanh', 'num_classes': 10, 'dropout_rate': None}
        net = VGG16(classifier_config)
    else:
        raise NotImplementedError
    return net(x, is_training)


@jax.jit
def loss_fn(params, x, y, is_training=True):
    logits = model.apply(params, x, is_training=is_training)
    # loss = optax.softmax_cross_entropy(logits, y)
    # return loss[0]
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


if __name__ == '__main__':
    argParser = get_arg_parser()
    args = argParser.parse_args()

    # Write config and log
    if not args.debug:
        timestamp = time.strftime('%b-%d-%Y-%H%M', time.localtime())
        folder_name = "{}-{}".format(timestamp, random.randint(0, 1000))
        print('* Results writing to: {}'.format(folder_name))
        result_path = os.path.join(args.result_path, folder_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        argparse_dict = vars(args)
        with open(os.path.join(result_path, "arguments.json"), "w") as f:
            json.dump(argparse_dict, f)
        train_log = os.path.join(result_path, 'train_log.txt')
        test_log = os.path.join(result_path, 'test_log.txt')

    # Load data
    if args.dataset == "cifar10":
        augmentations = [
            # transforms.Resize([224, 224]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            Cast(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        train_transform = transforms.Compose(augmentations + normalize)
        test_transform = transforms.Compose(normalize)

        train_dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        train_loader = NumpyLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            pin_memory=True,
        )
        test_loader = NumpyLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )
    elif args.dataset == "mnist":
        normalize = [
            Cast(),
            Normalize([0.5, ], [0.5, ]),
            AddChannelDim(),
        ]
        transform = transforms.Compose(normalize)

        mnist = MNIST(args.data_path, download=False, transform=transform)
        dataset_size = len(mnist)
        train_loader = NumpyLoader(
            mnist,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        mnist_test = MNIST(args.data_path, train=False, download=False, transform=transform)
        test_loader = NumpyLoader(
            mnist_test,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
    else:
        raise NotImplementedError

    N_train = len(train_loader.dataset)
    n_batch_steps = N_train // args.lot_size

    model = haiku.without_apply_rng(haiku.transform(net_fn))

    # Inits
    prng_seq = haiku.PRNGSequence(args.seed)
    optimizer = optax.sgd(learning_rate=args.lr, momentum=args.momentum)
    # dummy_input = next(iter(train_loader))[0]
    # dummy_input = dummy_input.cpu().detach().numpy()
    # dummy_input = jnp.asarray(dummy_input)
    # params = model.init(next(prng_seq), dummy_input)
    params = model.init(next(prng_seq), next(iter(train_loader))[0], is_training=True)
    opt_state = optimizer.init(params)
    privacy_accountant = RDPAccountant()

    Accuracy = []
    Epsilon = []
    Test_accuracy = []
    Test_epsilon = []

    prune_masks = {}
    prune_masks_tree = []
    prune_iter_counter = 0
    avg_sparcity = -1

    clip = args.clip
    noise_multiplier = args.noise_multiplier
    lot_size = args.lot_size

    mean_gradients_size_after, mean_gradients_size_before = None, None

    lot_idx_counter = 0

    for e in range(args.epochs):

        gradients = None
        correct_preds = 0
        total_preds = 0
        gradients_size = 0
        gradients_size_clipped = 0
        grads_norm_per_layer_sum = None
        grads_norm_clipped_per_layer_sum = None
        mean_gradients_size_list = []

        for i, batch in enumerate(train_loader):
            # Processing data format for jax
            batch_x, batch_y = batch
            # batch_x = batch_x.cpu().detach().numpy()
            # batch_y = batch_y.cpu().detach().numpy()
            # batch_x = jnp.asarray(batch_x)
            # batch_y = jnp.asarray(batch_y)

            # Prediction
            correct, total = predictions(params, batch_x, batch_y)
            correct_preds += correct
            total_preds += total

            # Compute gradient (forward + backward)
            loss, grads = get_loss_grads(params, batch_x, batch_y)

            # Clip (& Accumulate) gradient
            grads, grads_norm, grads_norm_clipped, grads_norm_per_layer, grads_norm_clipped_per_layer, \
                grads_min, grads_max, grads_clipped_min, grads_clipped_max = \
                clip_grads(grads, max_clipping_value=clip, prune_masks_tree=prune_masks_tree)
            grads_norm_per_layer = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_norm_per_layer)
            grads_norm_clipped_per_layer = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_norm_clipped_per_layer)
            grads_min = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_min)
            grads_max = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_max)
            grads_clipped_min = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_clipped_min)
            grads_clipped_max = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads_clipped_max)
            grads_min_dict = get_per_layer_grad_norm(grads_min)
            grads_max_dict = get_per_layer_grad_norm(grads_max)
            grads_clipped_min_dict = get_per_layer_grad_norm(grads_clipped_min)
            grads_clipped_max_dict = get_per_layer_grad_norm(grads_clipped_max)

            # print("Min; Max; ClipMin, ClipMax")
            # print(grads_min_dict.values())
            # print(grads_max_dict.values())
            # print(grads_clipped_min_dict.values())
            # print(grads_clipped_max_dict.values())

            if grads_norm_per_layer_sum is None:
                grads_norm_per_layer_sum = grads_norm_per_layer
                grads_norm_clipped_per_layer_sum = grads_norm_clipped_per_layer
            else:
                grads_norm_per_layer_sum = jax.tree_multimap(
                    lambda x, y: x+y, grads_norm_per_layer_sum, grads_norm_per_layer)
                grads_norm_clipped_per_layer_sum = jax.tree_multimap(
                    lambda x, y: x + y, grads_norm_clipped_per_layer_sum, grads_norm_clipped_per_layer)
            gradients_size += jnp.mean(grads_norm)
            gradients_size_clipped += jnp.mean(grads_norm_clipped)

            if gradients is None:
                grads = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), grads)
            else:
                grads = jax.tree_util.tree_multimap(
                    lambda x, y: x + jnp.sum(y, axis=0),
                    gradients, grads
                )
            gradients = grads

            if (i + 1) % (lot_size // args.batch_size) == 0:
                lot_idx_counter += 1

                mean_gradients_size = gradients_size / (lot_size // args.batch_size)
                # mean_gradients_size_list.append(mean_gradients_size)
                mean_gradients_size_clipped = gradients_size_clipped / (lot_size // args.batch_size)
                grads_norm_per_layer_sum = jax.tree_map(
                    lambda x: x / (lot_size // args.batch_size), grads_norm_per_layer_sum)
                grads_norm_clipped_per_layer_sum = jax.tree_map(
                    lambda x: x / (lot_size // args.batch_size), grads_norm_clipped_per_layer_sum)
                # grads_norm_per_layer_dict = get_per_layer_grad_norm(grads_norm_per_layer_sum)
                # grads_norm_clipped_per_layer_dict = get_per_layer_grad_norm(grads_norm_clipped_per_layer_sum)

                # Add noise
                gradients, grads_norm_noised, grads_norm_noised_per_layer = noise_grads(
                    gradients,
                    max_clipping_value=clip,
                    noise_multiplier=noise_multiplier,
                    lot_size=lot_size,
                    seed=next(prng_seq),
                    prune_masks_tree=prune_masks_tree
                )
                #  = get_per_layer_grad_norm(grads_norm_noised_per_layer)
                # Descent (update)
                params, opt_state = update(params, gradients, opt_state)
                # params = update_prune(params, prune_masks_tree)

                # # TMP: noise_schedulers
                if args.noise_scheduler == 'exp_decay':
                    noise_multiplier *= 0.9999
                elif args.noise_scheduler == 'exp_increase':
                    noise_multiplier *= 1.01
                elif args.noise_scheduler == 'exp_inc_dec':
                    if lot_idx_counter < 2000:
                        noise_multiplier *= 1.0001
                    else:
                        noise_multiplier *= 0.9999

                # TMP: steps
                # if lot_idx_counter == args.tmp_step:
                #     noise_multiplier -= 1
                if lot_idx_counter == args.tmp_step:
                    lot_size *= 2

                # Privacy Accountant
                privacy_accountant.step(noise_multiplier=noise_multiplier,
                                        sample_rate=lot_size / N_train)
                eps = privacy_accountant.get_epsilon(delta=args.delta)

                # Logging
                print('Epoch: {}, Batch: {}, Acc = {:.3f}, Eps = {:.3f}, Clip = {:.3f}, '
                      'NM = {:.3f}, grad_norm = {:.3f}, mean_loss = {:.3f}, std_loss = {:.3f}'.format(
                    e, i, float(correct_preds / total_preds), eps, clip, noise_multiplier, float(mean_gradients_size),
                    jnp.mean(loss), jnp.std(loss)
                ))
                Accuracy.append(correct_preds / total_preds)
                Epsilon.append(eps)
                if not args.debug:
                    log_items = [
                        e, i, float(correct_preds / total_preds), eps, avg_sparcity, clip, noise_multiplier, lot_size,
                        float(mean_gradients_size), float(mean_gradients_size_clipped), float(grads_norm_noised),
                        jnp.mean(loss), jnp.std(loss)
                    ]
                    log_str = '{:.3f} | ' * (len(log_items) - 1) + '{:.3f}'
                    formatted_log = log_str.format(*log_items)
                    # formatted_log = formatted_log + " | " + str(grads_norm_per_layer_dict) + \
                    #                 " | " + str(grads_norm_clipped_per_layer_dict) + \
                    #                 " | " + str(grads_norm_noised_per_layer_dict)
                    with open(train_log, 'a+') as f:
                        f.write(formatted_log)
                        f.write('\n')

                # Reset
                gradients = None
                correct_preds = 0
                total_preds = 0
                gradients_size = 0
                gradients_size_clipped = 0
                grads_norm_per_layer_sum = None
                grads_norm_clipped_per_layer_sum = None

        # evaluate test accuracy
        correct_preds = 0
        total_preds = 0
        for test_i, test_batch in enumerate(test_loader):
            # Processing data format for jax
            test_batch_x, test_batch_y = test_batch
            # test_batch_x = test_batch_x.cpu().detach().numpy()
            # test_batch_y = test_batch_y.cpu().detach().numpy()
            # test_batch_x = jnp.asarray(test_batch_x)
            # test_batch_y = jnp.asarray(test_batch_y)
            # test pred
            correct, total = predictions(params, test_batch_x, test_batch_y)
            correct_preds += correct
            total_preds += total
        print('Test epoch {}, Test Acc = {:.3f}, Cur_eps = {:.3f}'.format(
            e, correct_preds / total_preds, eps
        ))
        Test_accuracy.append(correct_preds / total_preds)
        Test_epsilon.append(eps)

        if not args.debug:
            log_items = [
                e, correct_preds / total_preds, eps
            ]
            log_str = '{:.3f}_' * (len(log_items) - 1) + '{:.3f}'
            formatted_log = log_str.format(*log_items)
            with open(test_log, 'a+') as f:
                f.write(formatted_log)
                f.write('\n')
