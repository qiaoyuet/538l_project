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
# from prune.prune import L1Unstructured
# from haiku._src.data_structures import FlatMap
import pickle

from util.dataloader import Normalize, AddChannelDim, Cast, NumpyLoader
from util.dp_utils import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

NUM_CLASSES = 10

def net_fn(x, is_training=True):
    if args.model == 'resnet9':
        net = ResNet9(num_classes=NUM_CLASSES)
        # net = ResNet101(num_classes=NUM_CLASSES)
    elif args.model == 'cnn_small':
        classifier_config = {'activation': 'tanh', 'num_classes': NUM_CLASSES, 'dropout_rate': None}
        net = CNNSmall(classifier_config)
    elif args.model == 'cnn_med':
        classifier_config = {'activation': 'tanh', 'num_classes': NUM_CLASSES, 'dropout_rate': None}
        net = CNNMed(classifier_config)
    elif args.model == 'vgg16':
        classifier_config = {'activation': 'tanh', 'num_classes': NUM_CLASSES, 'dropout_rate': 0.5}
        # classifier_config = {'activation': 'tanh', 'num_classes': NUM_CLASSES, 'dropout_rate': None}
        net = VGG16(classifier_config)
    else:
        raise NotImplementedError
    return net(x, is_training)


# @jax.jit
# def loss_fn(params, x, y, is_training=True):
#     logits = model.apply(params, x, is_training=is_training)
#     # loss = optax.softmax_cross_entropy(logits, y)
#     # return loss[0]
#     labels = jax.nn.one_hot(y, 10)
#     softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
#     return softmax_xent


@jax.jit
def loss_fn(trainable_params, non_trainable_params, x, y, is_training=True):
    params = haiku.data_structures.merge(trainable_params, non_trainable_params)
    logits = model.apply(params, x, is_training=is_training)
    labels = jax.nn.one_hot(y, NUM_CLASSES)
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    return softmax_xent


# @jax.jit
# def update(params, grads, opt_state):
#     updates, opt_state = optimizer.update(grads, opt_state)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, opt_state

def sgd_step(params, grads, *, lr):
    return jax.tree_map(lambda p, g: p - g * lr, params, grads)

@jax.jit
def update(trainable_params, trainable_params_grads):
    trainable_params = sgd_step(trainable_params, trainable_params_grads, lr=args.lr)
    return trainable_params

@jax.jit
def predictions(params, x, y):
    pred_y = model.apply(params, x, is_training=False)
    correct = jnp.sum(jnp.argmax(pred_y, axis=-1) == y)
    return correct, x.shape[0]


# @jax.jit
# def get_loss_grads(params, x, y):
#     x = jnp.expand_dims(x, axis=1)
#     y = jnp.expand_dims(y, axis=1)
#     get_value_grads = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(None, 0, 0))
#     loss, grads = get_value_grads(params, x, y)
#     return loss, grads

@jax.jit
def get_loss_grads(trainable_params, non_trainable_params, x, y):
    x = jnp.expand_dims(x, axis=1)
    y = jnp.expand_dims(y, axis=1)
    # Take derivative with respect to trainable parameters only
    get_value_grads = jax.vmap(jax.value_and_grad(loss_fn, argnums=0), in_axes=(None, None, 0, 0))
    loss, grads = get_value_grads(trainable_params, non_trainable_params, x, y)
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
    params = model.init(next(prng_seq), next(iter(train_loader))[0], is_training=True)
    opt_state = optimizer.init(params)
    privacy_accountant = RDPAccountant()

    Accuracy = []
    Epsilon = []
    Test_accuracy = []
    Test_epsilon = []

    clip = args.clip
    noise_multiplier = args.noise_multiplier
    lot_size = args.lot_size

    mean_gradients_size_after, mean_gradients_size_before = None, None

    lot_idx_counter = 0

    trainable_params = params
    non_trainable_params = []
    # trainable_params, non_trainable_params = haiku.data_structures.partition(
    #     lambda m, n, p: (m != 'cnn_small/linear') & (m != 'cnn_small/conv2_d'), params)
    print("trainable:", list(trainable_params))
    print("non_trainable:", list(non_trainable_params))

    # TMP:
    old_grad_norms_by_layer = []
    cur_grad_norms_by_layer = []

    for e in range(args.epochs):

        trainable_params_gradients = None
        correct_preds = 0
        total_preds = 0
        gradients_size = 0
        tmp_gradients_size_1 = 0
        tmp_gradients_size_2 = 0
        tmp_gradients_size_3 = 0
        tmp_gradients_size_4 = 0
        tmp_gradients_size_5 = 0
        gradients_size_clipped = 0
        gradients_size_max = 0
        gradients_size_clipped_max = 0
        grads_norm_per_layer_sum = None
        grads_norm_clipped_per_layer_sum = None
        mean_gradients_size_list = []

        if args.prune and (e > args.tmp_e1):
            # trainable_params, non_trainable_params = haiku.data_structures.partition(
            #     lambda m, n, p: (m != 'cnn_small/linear') & (m != 'cnn_small/conv2_d'), params)
            trainable_params, non_trainable_params = haiku.data_structures.partition(
                lambda m, n, p: m != 'cnn_med/conv2_d', params)
            print("trainable:", list(trainable_params))
            print("non_trainable:", list(non_trainable_params))

        if args.prune and (e > args.tmp_e2):
            trainable_params, non_trainable_params = haiku.data_structures.partition(
                lambda m, n, p: (m != 'cnn_med/conv2_d_1') and (m != 'cnn_med/conv2_d'), params)
            print("trainable:", list(trainable_params))
            print("non_trainable:", list(non_trainable_params))

        if args.prune and (e > args.tmp_e3):
            trainable_params, non_trainable_params = haiku.data_structures.partition(
                lambda m, n, p: (m != 'cnn_med/conv2_d_1') and (m != 'cnn_med/conv2_d') and (m != 'cnn_med/conv2_d_2'), params)
            print("trainable:", list(trainable_params))
            print("non_trainable:", list(non_trainable_params))

        if args.prune and (e > args.tmp_e4):
            trainable_params, non_trainable_params = haiku.data_structures.partition(
                lambda m, n, p: (m != 'cnn_med/conv2_d_1') and (m != 'cnn_med/conv2_d') and (m != 'cnn_med/conv2_d_2') and (m != 'cnn_med/conv2_d_3'), params)
            print("trainable:", list(trainable_params))
            print("non_trainable:", list(non_trainable_params))

        for i, batch in enumerate(train_loader):
            # Processing data format for jax
            batch_x, batch_y = batch

            # Prediction
            params = haiku.data_structures.merge(trainable_params, non_trainable_params)
            correct, total = predictions(params, batch_x, batch_y)
            correct_preds += correct
            total_preds += total

            # Compute gradient (forward + backward)
            loss, trainable_params_grads = get_loss_grads(trainable_params, non_trainable_params, batch_x, batch_y)
            # TMP
            # _, all_params_grads = get_loss_grads(params, non_trainable_params, batch_x, batch_y)
            if 'cnn_med/conv2_d' in trainable_params_grads.keys():
                tmp_grads_norm1 = jnp.sqrt(jax.tree_util.tree_reduce(
                    lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                    trainable_params_grads['cnn_med/conv2_d'], 0
                ))
            else:
                tmp_grads_norm1 = 0

            if 'cnn_med/conv2_d_1' in trainable_params_grads.keys():
                tmp_grads_norm2 = jnp.sqrt(jax.tree_util.tree_reduce(
                    lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                    trainable_params_grads['cnn_med/conv2_d_1'], 0
                ))
            else:
                tmp_grads_norm2 = 0

            if 'cnn_med/conv2_d_2' in trainable_params_grads.keys():
                tmp_grads_norm3 = jnp.sqrt(jax.tree_util.tree_reduce(
                    lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                    trainable_params_grads['cnn_med/conv2_d_2'], 0
                ))
            else:
                tmp_grads_norm3 = 0

            if 'cnn_med/conv2_d_3' in trainable_params_grads.keys():
                tmp_grads_norm4 = jnp.sqrt(jax.tree_util.tree_reduce(
                    lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                    trainable_params_grads['cnn_med/conv2_d_3'], 0
                ))
            else:
                tmp_grads_norm4 = 0

            if 'cnn_med/conv2_d_4' in trainable_params_grads.keys():
                tmp_grads_norm5 = jnp.sqrt(jax.tree_util.tree_reduce(
                    lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                    trainable_params_grads['cnn_med/conv2_d_4'], 0
                ))
            else:
                tmp_grads_norm5 = 0

            # # TMP
            # cur_grad_norms_by_layer.extend([tmp_grads_norm1, tmp_grads_norm2, tmp_grads_norm3, tmp_grads_norm4, tmp_grads_norm5])
            # if len(old_grad_norms_by_layer) !=0:
            #     grad_norm_diff = np.array(cur_grad_norms_by_layer) - np.array(old_grad_norms_by_layer)
            # old_grad_norms_by_layer = [tmp_grads_norm1, tmp_grads_norm2, tmp_grads_norm3, tmp_grads_norm4, tmp_grads_norm5]

            # Clip (& Accumulate) gradient
            trainable_params_grads, grads_norm, grads_norm_clipped, grads_norm_per_layer, grads_norm_clipped_per_layer, \
                grads_min, grads_max, grads_clipped_min, grads_clipped_max = \
                clip_grads(trainable_params_grads, max_clipping_value=clip, prune_masks_tree=[])
            # # TMP
            # _, grads_norm_orig, _, _, _, _, _, _, _ = \
            #     clip_grads(all_params_grads, max_clipping_value=clip, prune_masks_tree=[])
            # print(jnp.mean(grads_norm), jnp.mean(grads_norm_orig))

            gradients_size += jnp.mean(grads_norm)
            gradients_size_clipped += jnp.mean(grads_norm_clipped)
            gradients_size_max += jnp.max(grads_norm)
            gradients_size_clipped_max += jnp.max(grads_norm_clipped)
            tmp_gradients_size_1 += jnp.mean(tmp_grads_norm1)
            tmp_gradients_size_2 += jnp.mean(tmp_grads_norm2)
            tmp_gradients_size_3 += jnp.mean(tmp_grads_norm3)
            tmp_gradients_size_4 += jnp.mean(tmp_grads_norm4)
            tmp_gradients_size_5 += jnp.mean(tmp_grads_norm5)

            if trainable_params_gradients is None:
                trainable_params_grads = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), trainable_params_grads)
            else:
                trainable_params_grads = jax.tree_util.tree_multimap(
                    lambda x, y: x + jnp.sum(y, axis=0),
                    trainable_params_gradients, trainable_params_grads
                )
            trainable_params_gradients = trainable_params_grads

            if (i + 1) % (lot_size // args.batch_size) == 0:
                lot_idx_counter += 1
                mean_gradients_size = gradients_size / (lot_size // args.batch_size)
                mean_gradients_size_clipped = gradients_size_clipped / (lot_size // args.batch_size)
                max_gradients_size = gradients_size_max / (lot_size // args.batch_size)
                max_gradients_size_clipped = gradients_size_clipped_max / (lot_size // args.batch_size)
                mean_tmp_gradients_size_1 = tmp_gradients_size_1 / (lot_size // args.batch_size)
                mean_tmp_gradients_size_2 = tmp_gradients_size_2 / (lot_size // args.batch_size)
                mean_tmp_gradients_size_3 = tmp_gradients_size_3 / (lot_size // args.batch_size)
                mean_tmp_gradients_size_4 = tmp_gradients_size_4 / (lot_size // args.batch_size)
                mean_tmp_gradients_size_5 = tmp_gradients_size_5 / (lot_size // args.batch_size)

                # TMP:
                if not args.debug:
                    if i == (args.lot_size / args.batch_size) - 1:
                        tmp_dict = {}
                        for layer_name in trainable_params_gradients.keys():
                            tmp_name = layer_name.split('/')[-1]
                            if ('conv' in tmp_name) or ('linear' in tmp_name):
                                tmp_dict[layer_name] = np.array(trainable_params_gradients[layer_name]['w'])
                        tmp_out_name = os.path.join(result_path, "grads_{}_{}.pkl".format(e, i))
                        with open(tmp_out_name, "wb") as tmp_output:
                            pickle.dump(tmp_dict, tmp_output)

                # Add noise
                trainable_params_gradients, grads_norm_noised, grads_norm_noised_per_layer = noise_grads(
                    trainable_params_gradients,
                    max_clipping_value=clip,
                    noise_multiplier=noise_multiplier,
                    lot_size=lot_size,
                    seed=next(prng_seq),
                    prune_masks_tree=[]
                )
                # Descent (update)
                # params, opt_state = update(params, gradients, opt_state)
                trainable_params = update(trainable_params, trainable_params_gradients)

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
                        e, i, float(correct_preds / total_preds), eps, clip, noise_multiplier, lot_size,
                        float(mean_gradients_size), float(mean_gradients_size_clipped),
                        float(max_gradients_size), float(max_gradients_size_clipped),
                        float(grads_norm_noised),
                        jnp.mean(loss), jnp.std(loss),
                        float(mean_tmp_gradients_size_1), float(mean_tmp_gradients_size_2),
                        float(mean_tmp_gradients_size_3), float(mean_tmp_gradients_size_4),
                        float(mean_tmp_gradients_size_5),
                    ]
                    log_str = '{:.3f} | ' * (len(log_items) - 1) + '{:.3f}'
                    formatted_log = log_str.format(*log_items)
                    with open(train_log, 'a+') as f:
                        f.write(formatted_log)
                        f.write('\n')

                # Reset
                gradients = None
                correct_preds = 0
                total_preds = 0
                gradients_size = 0
                gradients_size_clipped = 0
                gradients_size_max = 0
                gradients_size_clipped_max = 0
                grads_norm_per_layer_sum = None
                grads_norm_clipped_per_layer_sum = None
                tmp_gradients_size_1 = 0
                tmp_gradients_size_2 = 0
                tmp_gradients_size_3 = 0
                tmp_gradients_size_4 = 0
                tmp_gradients_size_5 = 0

        # evaluate test accuracy
        correct_preds = 0
        total_preds = 0
        for test_i, test_batch in enumerate(test_loader):
            # Processing data format for jax
            test_batch_x, test_batch_y = test_batch
            # test pred
            params = haiku.data_structures.merge(trainable_params, non_trainable_params)
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
