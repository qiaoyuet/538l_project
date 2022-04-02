# 538l_project

#### Bash example

Normal dpsgd:
```python
python dpsgd_jax.py --lot_size 32 --batch_size 8 --data_path /home/qiaoyuet/project/cifar10 --epochs 5
```
With prune:
```python
python dpsgd_jax.py --lot_size 32 --batch_size 8 --data_path /home/qiaoyuet/project/cifar10 --epochs 5 --prune
```

#### Instructions for running pytorch version
dpsgd and pruning are enabled by default in the pytorch version.

To disable dpsgd run it like so:

``` bash
python prune/mnist_dp_pytorch.py --no-dpsgd
```

To disable pruning run it like so:

``` bash
python prune/mnist_dp_pytorch.py --no-prune
```

`

`
