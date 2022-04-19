from abc import ABC
from functools import reduce
import numpy as np
import jax.numpy as jnp
import jax


class BasePrune(ABC):
    def __init__(self):
        pass

    def compute_mask(self, t, mask, layer):
        pass

    def apply_mask(self, t, mask):
        pass


class L1Unstructured(BasePrune):
    """
    Prune units in a tensor by zeroing out the ones with the lowest L1-norm (in absolute value).
    """
    def __init__(self, conv_amount, linear_amount):
        super().__init__()
        self.conv_amount = conv_amount
        self.linear_amount = linear_amount

    def compute_mask(self, t, default_mask, layer_type):
        if default_mask is not None:
            # just for outputting statistics
            tmp_t = jnp.multiply(t, default_mask)
            pre_q_25, pre_q_50, pre_q_75 = np.quantile(np.abs(tmp_t), (0.25, 0.50, 0.75))
            pre_mean, pre_std = np.mean(np.abs(tmp_t)), np.std(np.abs(tmp_t))
            # trick: flip 0 and 1 and making masked position a really large value to exclude already masked weights
            tmp_scaled_default_mask = (default_mask - 1) * (-1) * 10e6
            t = jnp.add(t, tmp_scaled_default_mask)
        else:
            pre_q_25, pre_q_50, pre_q_75 = np.quantile(np.abs(t), (0.25, 0.50, 0.75))
            pre_mean, pre_std = np.mean(np.abs(t)), np.std(np.abs(t))
        # get the amount of units in a weight tensor
        tensor_size = t.shape
        tensor_nelement = reduce(lambda x, y: x*y, tensor_size)
        # get amount to prune
        if layer_type == 'conv':
            nparams_toprune = int(round(self.conv_amount * tensor_nelement))
        elif layer_type == 'linear':
            nparams_toprune = int(round(self.linear_amount * tensor_nelement))
        else:
            raise NotImplementedError
        # get mask
        mask_flat = np.ones(tensor_size).flatten()
        if nparams_toprune != 0:
            tmp_t = np.array(t).flatten()
            ind = np.argpartition(np.abs(tmp_t), nparams_toprune)[:nparams_toprune]  # jnp function not yet implemented
            tmp_after_pruned = np.abs(tmp_t)[ind]
            pruned_max = np.max(tmp_after_pruned)  # pruned threshold
            post_q_25, post_q_50, post_q_75 = np.quantile(tmp_after_pruned, (0.25, 0.50, 0.75))
            post_mean, post_std = np.mean(tmp_after_pruned), np.std(tmp_after_pruned)
            mask_flat[ind] = 0
            sparcity = np.count_nonzero(mask_flat==0)/len(mask_flat)
        else:
            raise ValueError()
        mask = mask_flat.reshape(tensor_size)
        mask = jnp.array(mask)

        return mask, sparcity, \
                (pre_q_25, pre_q_50, pre_q_75, pre_mean, pre_std,
                    post_q_25, post_q_50, post_q_75, post_mean, post_std, pruned_max)

    def apply_mask(self, t, mask):
        return jax.tree_multimap(lambda p, u: jnp.asarray(p * u).astype(jnp.asarray(p).dtype), t, mask)
