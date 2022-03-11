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

from arguments import get_arg_parser

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, epoch, device, privacy_engine):
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

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(args.delta)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta})"
                )


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Init privacy accountant
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.clip,
    )

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device, privacy_engine)
        if epoch % args.test_every_n == 0:
            top1_acc = test(model, test_loader, device)


if __name__ == "__main__":
    argParser = get_arg_parser()
    args = argParser.parse_args()
    main(args)