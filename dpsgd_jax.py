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
from model.cnn import CNNSmall, CNNMed
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
        classifier_config = {'activation': 'tanh', 'num_classes': 10}
        net = CNNSmall(classifier_config)
    elif args.model == 'cnn_med':
        classifier_config = {'activation': 'tanh', 'num_classes': 10}
        net = CNNMed(classifier_config)
    else:
        raise NotImplementedError
    return net(x, is_training)


@jax.jit
def loss_fn(params, x, y, is_training=True):
    logits = model.apply(params, x, is_training=is_training)
    loss = optax.softmax_cross_entropy(logits, y)
    return loss[0]
    # labels = jax.nn.one_hot(y, 10)
    # softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    # return softmax_xent


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

            if (i + 1) % (args.lot_size // args.batch_size) == 0:
                lot_idx_counter += 1
                mean_gradients_size = gradients_size / (args.lot_size // args.batch_size)
                # mean_gradients_size_list.append(mean_gradients_size)
                mean_gradients_size_clipped = gradients_size_clipped / (args.lot_size // args.batch_size)
                grads_norm_per_layer_sum = jax.tree_map(
                    lambda x: x / (args.lot_size // args.batch_size), grads_norm_per_layer_sum)
                grads_norm_clipped_per_layer_sum = jax.tree_map(
                    lambda x: x / (args.lot_size // args.batch_size), grads_norm_clipped_per_layer_sum)
                # grads_norm_per_layer_dict = get_per_layer_grad_norm(grads_norm_per_layer_sum)
                # grads_norm_clipped_per_layer_dict = get_per_layer_grad_norm(grads_norm_clipped_per_layer_sum)

                # Add noise
                gradients, grads_norm_noised, grads_norm_noised_per_layer = noise_grads(
                    gradients,
                    max_clipping_value=clip,
                    noise_multiplier=noise_multiplier,
                    lot_size=args.lot_size,
                    seed=next(prng_seq),
                    prune_masks_tree=prune_masks_tree
                )
                #  = get_per_layer_grad_norm(grads_norm_noised_per_layer)
                # Descent (update)
                params, opt_state = update(params, gradients, opt_state)
                # params = update_prune(params, prune_masks_tree)

                # TMP: noise_schedulers
                if args.noise_scheduler == 'exp_decay':
                    noise_multiplier *= 0.99
                elif args.noise_scheduler == 'exp_increase':
                    noise_multiplier *= 1.01

                # pruning
                # FIXME: lot_idx_counter or e
                if args.prune and \
                        (lot_idx_counter >= args.prune_after_n) and \
                        ((lot_idx_counter - args.prune_after_n) % (args.num_train_per_prune + 1) == 0) and \
                        (lot_idx_counter <= args.no_prune_after_n):

                    # Pruning
                    print('Pruning step')
                    prune_masks_tree = []
                    prune_iter_counter += 1
                    if (args.restore_every_n != -1) and (prune_iter_counter % args.restore_every_n == 0):
                        prune_masks = {}
                    pruned_params = []
                    sparcities = []
                    prunr_stats_list = []
                    prune_method = L1Unstructured(conv_amount=args.conv2d_prune_amount,
                                                  linear_amount=args.linear_prune_amount)
                    for module_name in list(params.keys()):
                        tmp_name = module_name.split('/')[-1]
                        if ('conv' in tmp_name) or ('linear' in tmp_name):
                            tmp_param = params[module_name]['w']
                            if module_name not in prune_masks:
                                default_mask = None
                            else:
                                default_mask = prune_masks[module_name]

                            if 'conv' in tmp_name:
                                prune_mask, sparcity, prune_stats = prune_method.compute_mask(
                                    t=tmp_param, default_mask=default_mask, layer_type='conv'
                                )
                            elif 'linear' in tmp_name:
                                prune_mask, sparcity, prune_stats = prune_method.compute_mask(
                                    t=tmp_param, default_mask=default_mask, layer_type='linear'
                                )
                            else:
                                raise NotImplementedError

                            prunr_stats_list.append(prune_stats)

                            if module_name not in prune_masks:
                                tmp_mask = prune_mask
                            else:
                                tmp_mask = jnp.multiply(prune_masks[module_name], prune_mask)
                            if 'b' in params[module_name].keys():
                                tmp_param_bias = params[module_name]['b']
                                tmp_prune_masks_tree = FlatMap(dict(w=tmp_mask,
                                                                    b=jnp.ones_like(tmp_param_bias)))
                            else:
                                tmp_prune_masks_tree = FlatMap(dict(w=tmp_mask))
                            tmp_mask_flat = np.array(tmp_mask).flatten()
                            cur_sparcity = np.count_nonzero(tmp_mask_flat == 0) / len(tmp_mask_flat)
                            sparcities.append(cur_sparcity)
                            prune_masks[module_name] = tmp_mask
                            apply_prune = prune_method.apply_mask(tmp_param, prune_masks[module_name])

                            if 'b' in params[module_name].keys():
                                pruned_param = FlatMap(dict(w=apply_prune, b=params[module_name]['b']))
                            else:
                                pruned_param = FlatMap(dict(w=apply_prune))
                        elif 'b' in tmp_name:
                            pruned_param = params[module_name]
                            sparcity = 0
                            tmp_prune_masks_tree = FlatMap(dict(b=jnp.ones_like(params[module_name]['b'])))
                        elif 'scale' in tmp_name:
                            pruned_param = params[module_name]
                            sparcity = 0
                            tmp_prune_masks_tree = FlatMap(dict(s=jnp.ones_like(params[module_name]['s'])))
                        elif 'logits' in tmp_name:
                            pruned_param = params[module_name]
                            sparcity = 0
                            tmp_prune_masks_tree = FlatMap(dict(w=jnp.ones_like(params[module_name]['w']),
                                                                b=jnp.ones_like(params[module_name]['b'])))
                        else:
                            raise NotImplementedError
                        pruned_params.append(pruned_param)
                        prune_masks_tree.append(tmp_prune_masks_tree)
                    params = update_prune(params, pruned_params)
                    prune_masks_tree = update_prune(params, prune_masks_tree)
                    avg_sparcity = np.mean(np.array(sparcities))
                    prunr_stats_list_array = np.array([np.array(tmp_stats) for tmp_stats in prunr_stats_list])
                    avg_pre_q_25, avg_pre_q_50, avg_pre_q_75, avg_pre_mean, avg_pre_std, \
                    avg_post_q_25, avg_post_q_50, avg_post_q_75, avg_post_mean, avg_post_std, avg_pruned_max = \
                        np.mean(prunr_stats_list_array, axis=0)

                    if args.rescale_type_1 or args.rescale_type_2:
                        # FIXME: protect priv for coef?
                        tmp_batch = next(iter(train_loader))
                        batch_x, batch_y = tmp_batch
                        _, tmp_grads = get_loss_grads(params, batch_x, batch_y)

                        tmp_grads_pruned = jax.tree_multimap(
                            lambda x, y: x * jnp.repeat(jnp.expand_dims(y, axis=0), repeats=x.shape[0], axis=0),
                            tmp_grads, prune_masks_tree
                        )

                        grads_norm_old = jnp.sqrt(jax.tree_util.tree_reduce(
                            lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                            tmp_grads, 0
                        ))

                        grads_norm_new = jnp.sqrt(jax.tree_util.tree_reduce(
                            lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
                            tmp_grads_pruned, 0
                        ))

                        reduced_coef = grads_norm_new / grads_norm_old

                        if args.rescale_type_1:
                            clip *= float(jnp.mean(grads_norm_new / grads_norm_old))
                        if args.rescale_type_2:
                            noise_multiplier /= float(jnp.mean(grads_norm_new / grads_norm_old))
                else:
                    # print('Regular training step')
                    # avg_sparcity = avg_sparcity
                    avg_pre_q_25, avg_pre_q_50, avg_pre_q_75, avg_pre_mean, avg_pre_std, \
                    avg_post_q_25, avg_post_q_50, avg_post_q_75, avg_post_mean, avg_post_std, avg_pruned_max = \
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

                # Privacy Accountant
                privacy_accountant.step(noise_multiplier=noise_multiplier,
                                        sample_rate=args.lot_size / N_train)
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
                        e, i, float(correct_preds / total_preds), eps, avg_sparcity, clip, noise_multiplier,
                        float(mean_gradients_size), float(mean_gradients_size_clipped), float(grads_norm_noised),
                        avg_pre_q_25, avg_pre_q_50, avg_pre_q_75, avg_pre_mean, avg_pre_std,
                        avg_post_q_25, avg_post_q_50, avg_post_q_75, avg_post_mean, avg_post_std, avg_pruned_max,
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
