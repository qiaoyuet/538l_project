import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    train_group = parser.add_argument_group("train")
    train_group.add_argument('--seed', type=int, default=1024)
    train_group.add_argument('--lot_size', type=int, default=256)
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--epochs', type=int, default=100)
    train_group.add_argument('--lr', type=float, default=1e-2)
    train_group.add_argument('--momentum', type=float, default=0.9)
    train_group.add_argument('--debug', action='store_true')
    train_group.add_argument('--result_path', type=str, default="./results/")
    train_group.add_argument('--data_path', type=str)
    train_group.add_argument('--test_every_n', type=int, default=5)
    train_group.add_argument('--dataset', choices=["cifar10", "mnist"])
    train_group.add_argument('--model', choices=["resnet9", "cnn_small", 'cnn_med', 'vgg16'])
    train_group.add_argument('--noise_scheduler', choices=["exp_decay", "exp_increase", "exp_inc_dec"], default=None)
    train_group.add_argument('--tmp_step', type=int, default=-1)

    dp_group = parser.add_argument_group("dp")
    dp_group.add_argument('--clip', type=float, default=1.0)
    dp_group.add_argument('--noise_multiplier', type=float, default=1.0)
    dp_group.add_argument('--delta', type=float, default=1e-5)
    dp_group.add_argument('--epsilon', type=float, default=10)

    prune_group = parser.add_argument_group("prune")
    prune_group.add_argument('--prune', action='store_true')
    prune_group.add_argument('--prune_after_n', type=int, default=1)
    prune_group.add_argument('--num_train_per_prune', type=int, default=10)
    prune_group.add_argument('--prune_type', choices=["grouped", "local"], default="grouped")
    prune_group.add_argument('--conv2d_prune_amount', type=float, default=0.1)
    prune_group.add_argument('--linear_prune_amount', type=float, default=0.1)
    prune_group.add_argument('--restore_every_n', type=int, default=5)
    prune_group.add_argument('--rescale_type_1', action='store_true')
    prune_group.add_argument('--rescale_type_2', action='store_true')
    prune_group.add_argument('--no_prune_after_n', type=int, default=3)
    prune_group.add_argument('--tmp_e1', type=int, default=1000)
    prune_group.add_argument('--tmp_e2', type=int, default=1000)
    prune_group.add_argument('--tmp_e3', type=int, default=1000)
    prune_group.add_argument('--tmp_e4', type=int, default=1000)
    prune_group.add_argument('--tmp_e5', type=int, default=1000)

    return parser
