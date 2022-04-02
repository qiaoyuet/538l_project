import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import models
from opacus.validators import ModuleValidator
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
import torch.nn.utils.prune as prune

from arguments import get_arg_parser

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, epoch, device, privacy_engine, prune=False):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.batch_size,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            if prune:
                l1_reg = torch.tensor(0.).to(device)
                for module in model.modules():
                    mask = None
                    weight = None
                    for name, buffer in module.named_buffers():
                        if name == "weight_mask":
                            mask = buffer
                    for name, param in module.named_parameters():
                        if name == "weight_orig":
                            weight = param
                    # We usually only want to introduce sparsity to weights and prune weights.
                    # Do the same for bias if necessary.
                    if mask is not None and weight is not None:
                        l1_reg += torch.norm(mask * weight, 1)

                l1_regularization_strength = 0
                loss += l1_regularization_strength * l1_reg

            loss.backward()

            # calculate grad norm
            norm_sum_accumulator = 0
            tmp_min_accumulator = []
            tmp_max_accumulator = []
            for param in model.parameters():
                tmp_grad = param.grad
                tmp_min = np.amin(np.square(param.cpu().detach().numpy()))
                tmp_max = np.amax(np.square(param.cpu().detach().numpy()))
                tmp_min_accumulator.append(tmp_min)
                tmp_max_accumulator.append(tmp_max)
                tmp_sum = np.sum(np.square(tmp_grad.cpu().detach().numpy()))
                norm_sum_accumulator += tmp_sum
            grad_norm = np.sqrt(norm_sum_accumulator)
            grad_min = np.mean(tmp_min_accumulator)
            grad_max = np.mean(tmp_max_accumulator)

            optimizer.step()

            # calculate grad norm after clipping and noise (?)
            post_norm_sum_accumulator = 0
            post_tmp_min_accumulator = []
            post_tmp_max_accumulator = []
            for param in model.parameters():
                tmp_grad = param.grad
                tmp_min = np.amin(np.square(param.cpu().detach().numpy()))
                tmp_max = np.amax(np.square(param.cpu().detach().numpy()))
                post_tmp_min_accumulator.append(tmp_min)
                post_tmp_max_accumulator.append(tmp_max)
                tmp_sum = np.sum(np.square(tmp_grad.cpu().detach().numpy()))
                post_norm_sum_accumulator += tmp_sum
            post_grad_norm = np.sqrt(post_norm_sum_accumulator)
            post_grad_min = np.mean(post_tmp_min_accumulator)
            post_grad_max = np.mean(post_tmp_max_accumulator)

            if (i + 1) % 50 == 0:
                epsilon = privacy_engine.get_epsilon(args.delta)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta})"
                    f"Grad norm: {grad_norm:.3f} "
                    f"Grad min: {grad_min:.3f} "
                    f"Grad max: {grad_max:.3f} "
                    f"Post Grad norm: {post_grad_norm:.3f} "
                    f"Post Grad min: {post_grad_min:.3f} "
                    f"Post rad max: {post_grad_max:.3f} "
                )


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def train_with_prune(
        model, train_loader, optimizer, device, privacy_engine, epoch,
        prune_type, conv2d_prune_amount, linear_prune_amount, num_train_per_prune
):
    # Pruning
    if prune_type == 'grouped':
        # Global pruning
        # I would rather call it grouped pruning.
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=conv2d_prune_amount,
        )
    elif prune_type == 'local':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module,
                                      name="weight",
                                      amount=conv2d_prune_amount)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module,
                                      name="weight",
                                      amount=linear_prune_amount)
    else:
        raise NotImplementedError

    # TODO: eval?

    # Train
    for inside_epoch in range(num_train_per_prune):
        train(model, train_loader, optimizer, epoch, device, privacy_engine, prune=True)

    num_zeros, num_elements, sparsity = measure_global_sparsity(
        model,
        weight=True,
        bias=False,
        conv2d_use_mask=True,
        linear_use_mask=False)

    # TODO: logging


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        batch_size=args.lot_size,
    )

    test_dataset = CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.lot_size,
        shuffle=False,
    )

    # Load model
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model)  # replace batchnorm with groupnorm
    model = model.to(device)

    # Init optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Init privacy accountant
    privacy_engine = PrivacyEngine()
    # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     epochs=args.epochs,
    #     target_epsilon=args.epsilon,
    #     target_delta=args.delta,
    #     max_grad_norm=args.clip,
    # )
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.clip,
    )

    # Training loop
    if args.prune:
        train_with_prune(
            model, train_loader, optimizer, device, privacy_engine, epoch=args.epochs,
            prune_type=args.prune_type, conv2d_prune_amount=args.conv2d_prune_amount,
            linear_prune_amount=args.linear_prune_amount, num_train_per_prune=args.num_train_per_prune
        )
    else:
        for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
            train(model, train_loader, optimizer, epoch + 1, device, privacy_engine)
            if epoch % args.test_every_n == 0:
                top1_acc = test(model, test_loader, device)


if __name__ == "__main__":
    argParser = get_arg_parser()
    args = argParser.parse_args()
    main(args)