from abc import ABC
from functools import reduce
import numpy as np
import jax.numpy as jnp
import jax


class BasePrune(ABC):
    def __init__(self):
        pass

    def compute_mask(self, t):
        pass

    def apply_mask(self, t, mask):
        pass


class L1Unstructured(BasePrune):
    """
    Prune units in a tensor by zeroing out the ones with the lowest L1-norm (in absolute value).
    """
    def __init__(self, amount):
        super().__init__()
        self.amount = amount
        assert self.amount < 1.0, "Prune fraction cannot be smaller than 1."
        assert self.amount > 0.0, "Prune fraction cannot be negative."

    def compute_mask(self, t, default_mask):
        if default_mask:
            mask = jnp.ones_like(t)
        else:
            # get the amount of units in a weight tensor
            tensor_size = t.shape
            tensor_nelement = reduce(lambda x, y: x*y, tensor_size)
            # get amount to prune
            nparams_toprune = int(round(self.amount * tensor_nelement))
            # get mask
            mask_flat = np.ones(tensor_size).flatten()
            if nparams_toprune != 0:
                tmp_t = np.array(t).flatten()
                ind = np.argpartition(np.abs(tmp_t), nparams_toprune)[:nparams_toprune]  # jnp function not yet implemented
                mask_flat[ind] = 0
                sparcity = np.count_nonzero(mask_flat==0)/len(mask_flat)
            mask = mask_flat.reshape(tensor_size)
            mask = jnp.array(mask)

        return mask, sparcity

    def apply_mask(self, t, mask):
        return jax.tree_multimap(lambda p, u: jnp.asarray(p * u).astype(jnp.asarray(p).dtype), t, mask)
