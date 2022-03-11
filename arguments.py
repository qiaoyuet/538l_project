import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    train_group = parser.add_argument_group("train")
    train_group.add_argument('--seed', type=int, default=1024)
    train_group.add_argument('--lot_size', type=int, default=256)
    train_group.add_argument('--batch_size', type=int, default=16)
    train_group.add_argument('--epochs', type=int, default=200)
    train_group.add_argument('--lr', type=float, default=1e-2)
    train_group.add_argument('--momentum', type=float, default=0.9)
    train_group.add_argument('--debug', action='store_true')
    train_group.add_argument('--result_path', type=str, default="./results/")
    train_group.add_argument('--data_path', type=str)
    train_group.add_argument('--test_every_n', type=int, default=5)

    dp_group = parser.add_argument_group("dp")
    dp_group.add_argument('--clip', type=float, default=1.0)
    dp_group.add_argument('--noise_multiplier', type=float, default=1.0)
    dp_group.add_argument('--delta', type=float, default=1e-5)
    dp_group.add_argument('--epsilon', type=float, default=10)

    return parser
