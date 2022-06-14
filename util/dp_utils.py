import jax
import jax.numpy as jnp
import optax
from haiku._src.data_structures import FlatMap

@jax.jit
def clip_grads(grads, max_clipping_value, prune_masks_tree):
    if prune_masks_tree != []:
        grads = jax.tree_multimap(
            lambda x, y: x * jnp.repeat(jnp.expand_dims(y, axis=0), repeats=x.shape[0], axis=0),
            grads, prune_masks_tree)

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
def noise_grads(grads, max_clipping_value, noise_multiplier, lot_size, seed, prune_masks_tree):
    grads_flat, grads_treedef = jax.tree_flatten(grads)
    (*rngs,) = jax.random.split(seed, len(grads_flat))
    noised = []
    noise_z = []
    for g, r in zip(grads_flat, rngs):
        z = jax.random.normal(r, g.shape, g.dtype)
        noised.append(g + (max_clipping_value * noise_multiplier) * z)
        noise_z.append(z)
    noise_grads = jax.tree_unflatten(grads_treedef, noised)
    noise_grads = jax.tree_util.tree_map(lambda x: x / lot_size, noise_grads)
    noise_z_tree = jax.tree_unflatten(grads_treedef, noise_z)
    avg_noise = jax.tree_util.tree_map(lambda x: x / lot_size, noise_z_tree)

    if prune_masks_tree != []:
        noise_grads = jax.tree_multimap(lambda x, y: x * y, noise_grads, prune_masks_tree)

    grads_norm_noised = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda agg, x: agg + jnp.sum(jnp.reshape(jnp.square(x), (1, -1)), axis=-1),
        noise_grads, 0
    ))
    grads_norm_noised_per_layer = jax.tree_map(
        lambda x: jnp.sqrt(jnp.sum(jnp.reshape(jnp.square(x), (1, -1)), axis=-1)), noise_grads)

    return noise_grads, grads_norm_noised, grads_norm_noised_per_layer, avg_noise


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
        elif ('conv' in tmp_name) or ('linear' in tmp_name):
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
