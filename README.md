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
