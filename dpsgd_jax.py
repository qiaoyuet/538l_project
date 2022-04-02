import time
import random
import json
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
from arguments import get_arg_parser
from model.fixup_resnet import ResNet9
from prune.prune import L1Unstructured
from haiku._src.data_structures import FlatMap
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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

    grads_norm_per_layer = jax.tree_map(lambda x: jnp.sqrt(jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1)), grads)
    grads_min = jax.tree_map(lambda x: jnp.min(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1), grads)
    grads_max = jax.tree_map(lambda x: jnp.max(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1), grads)

    grads = jax.tree_util.tree_map(
        lambda x: x / jnp.maximum(
            1,
            jnp.expand_dims(grads_norm, axis=tuple(range(1, len(x.shape)))) / max_clipping_value
        ),
        grads
    )
    grads_norm_clipped = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1),
        grads, 0
    ))

    grads_norm_clipped_per_layer = jax.tree_map(lambda x: jnp.sqrt(jnp.sum(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1)), grads)
    grads_clipped_min = jax.tree_map(lambda x: jnp.min(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1), grads)
    grads_clipped_max = jax.tree_map(lambda x: jnp.max(jnp.reshape(jnp.square(x), (x.shape[0], -1)), axis=-1), grads)

    return grads, grads_norm, grads_norm_clipped, grads_norm_per_layer, grads_norm_clipped_per_layer, \
            grads_min, grads_max, grads_clipped_min, grads_clipped_max


@jax.jit
def noise_grads(grads, max_clipping_value, noise_multiplier, lot_size, seed):
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    (*rngs,) = jax.random.split(seed, len(grads_flat))
    noised = []
    # noise_z = []
    for g, r in zip(grads_flat, rngs):
        z = jax.random.normal(r, g.shape, g.dtype)
        noised.append(g + (max_clipping_value * noise_multiplier) * z)
        # noise_z.append(z)
    noise_grads = jax.tree_unflatten(grads_treedef, noised)
    noise_grads = jax.tree_util.tree_map(lambda x: x / lot_size, noise_grads)
    grads_norm_noised = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (1, -1)), axis=-1),
        noise_grads, 0
    ))
    grads_norm_noised_per_layer = jax.tree_map(
        lambda x: jnp.sqrt(jnp.sum(jnp.reshape(jnp.square(x), (1, -1)), axis=-1)), noise_grads)

    return noise_grads, grads_norm_noised, grads_norm_noised_per_layer


@jax.jit
def update_prune(params, pruned_params):
    tmp_keys = list(params.keys())
    assert len(tmp_keys) == len(pruned_params), "Lens differ."
    new_params = FlatMap(dict(zip(tmp_keys, pruned_params)))
    return new_params


def get_per_layer_grad_norm(grad_norm_tree):
    res_dict = {}
    # res_names = []
    # res_values = []
    for layer_name in grad_norm_tree.keys():
        # res_names.append(layer_name)
        tmp_name = layer_name.split('/')[-1]
        if 'b' in tmp_name:
            tmp_value = float(grad_norm_tree[layer_name]['b'])
        elif 'conv' in tmp_name:
            tmp_value = float(grad_norm_tree[layer_name]['w'])
        elif 'scale' in tmp_name:
            tmp_value = float(grad_norm_tree[layer_name]['s'])
        elif 'logits' in tmp_name:
            tmp_value = (float(grad_norm_tree[layer_name]['b']),
                         float(grad_norm_tree[layer_name]['w']))
        else:
            raise NotImplementedError('Unknown key.')
        res_dict[layer_name] = tmp_value
        # res_values.append(tmp_value)
    return res_dict
    # return res_names, res_values


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

    prune_masks = {}
    prune_iter_counter = 0

    for e in range(args.epochs):

        if args.prune and (e > args.prune_after_n - 1) and ((e - args.prune_after_n) % (args.num_train_per_prune + 1) == 0):
            # Pruning
            print('Pruning step')
            prune_iter_counter += 1
            if prune_iter_counter % args.restore_every_n == 0:
                prune_masks = {}
            pruned_params = []
            sparcities = []
            prune_method = L1Unstructured(amount=args.conv2d_prune_amount)
            for module_name in list(params.keys()):
                tmp_name = module_name.split('/')[-1]
                if 'conv' in tmp_name:
                    tmp_param = params[module_name]['w']
                    prune_mask, sparcity = prune_method.compute_mask(t=tmp_param, default_mask=False)
                    if module_name not in prune_masks:
                        tmp_mask = prune_mask
                    else:
                        tmp_mask = jnp.multiply(prune_masks[module_name], prune_mask)
                    tmp_mask_flat = np.array(tmp_mask).flatten()
                    cur_sparcity = np.count_nonzero(tmp_mask_flat == 0) / len(tmp_mask_flat)
                    sparcities.append(cur_sparcity)
                    prune_masks[module_name] = tmp_mask
                    apply_prune = prune_method.apply_mask(tmp_param, prune_masks[module_name])
                    pruned_param = FlatMap(dict(w=apply_prune))
                elif ('b' in tmp_name) or ('scale' in tmp_name) or ('logits' in tmp_name):
                    pruned_param = params[module_name]
                    sparcity = 0
                else:
                    raise NotImplementedError
                pruned_params.append(pruned_param)
            params = update_prune(params, pruned_params)
            avg_sparcity = np.mean(np.array(sparcities))
        else:
            print('Regular training step')
            avg_sparcity = -1

        gradients = None
        correct_preds = 0
        total_preds = 0
        gradients_size = 0
        gradients_size_clipped = 0
        grads_norm_per_layer_sum = None
        grads_norm_clipped_per_layer_sum = None
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
            grads, grads_norm, grads_norm_clipped, grads_norm_per_layer, grads_norm_clipped_per_layer, \
                grads_min, grads_max, grads_clipped_min, grads_clipped_max = \
                clip_grads(grads, max_clipping_value=args.clip)
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

            print("Min; Max; ClipMin, ClipMax")
            print(grads_min_dict.values())
            print(grads_max_dict.values())
            print(grads_clipped_min_dict.values())
            print(grads_clipped_max_dict.values())

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
                mean_gradients_size = gradients_size / (args.lot_size // args.batch_size)
                mean_gradients_size_clipped = gradients_size_clipped / (args.lot_size // args.batch_size)
                grads_norm_per_layer_sum = jax.tree_map(
                    lambda x: x / (args.lot_size // args.batch_size), grads_norm_per_layer_sum)
                grads_norm_clipped_per_layer_sum = jax.tree_map(
                    lambda x: x / (args.lot_size // args.batch_size), grads_norm_clipped_per_layer_sum)
                grads_norm_per_layer_dict = get_per_layer_grad_norm(grads_norm_per_layer_sum)
                grads_norm_clipped_per_layer_dict = get_per_layer_grad_norm(grads_norm_clipped_per_layer_sum)

                # Add noise
                gradients, grads_norm_noised, grads_norm_noised_per_layer = noise_grads(
                    gradients,
                    max_clipping_value=args.clip,
                    noise_multiplier=args.noise_multiplier,
                    lot_size=args.lot_size,
                    seed=next(prng_seq)
                )
                grads_norm_noised_per_layer_dict = get_per_layer_grad_norm(grads_norm_noised_per_layer)
                # Descent (update)
                params, opt_state = update(params, gradients, opt_state)
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
                if not args.debug:
                    log_items = [
                        e, i, float(correct_preds / total_preds), eps, avg_sparcity,
                        float(mean_gradients_size), float(mean_gradients_size_clipped), float(grads_norm_noised),
                    ]
                    log_str = '{:.3f} | ' * (len(log_items) - 1) + '{:.3f}'
                    formatted_log = log_str.format(*log_items)
                    formatted_log = formatted_log + " | " + str(grads_norm_per_layer_dict) + \
                                    " | " + str(grads_norm_clipped_per_layer_dict) + \
                                    " | " + str(grads_norm_noised_per_layer_dict)
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

        if not args.debug:
            log_items = [
                e, correct_preds / total_preds, eps
            ]
            log_str = '{:.3f}_' * (len(log_items) - 1) + '{:.3f}'
            formatted_log = log_str.format(*log_items)
            with open(test_log, 'a+') as f:
                f.write(formatted_log)
                f.write('\n')
