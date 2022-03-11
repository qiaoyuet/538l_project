"""Resnet."""

import types
from typing import Mapping, Optional, Sequence, Union, Any

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from haiku._src import utils


class Scale(hk.Module):
    """Scales the inputs.
    """

    def __init__(
        self,
        output_size: Optional[Sequence[int]] = None,
        scale_dims: Optional[Sequence[int]] = None,
        s_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """Constructs a ``Scale`` module that supports broadcasting.
        Args:
          output_size: Output size (output shape without batch dimension). If
            ``output_size`` is left as `None`, the size will be directly inferred by
            the input.
          scale_dims: Sequence of which dimensions to retain from the input shape
            when constructing the bias. The remaining dimensions will be broadcast
            over (given size of 1), and leading dimensions will be removed
            completely. See class doc for examples.
          s_init: Optional initializer for the bias. Default to zeros.
          name: Name of the module.
        """
        super().__init__(name=name)
        self.output_size = output_size
        self.scale_dims = scale_dims
        self.s_init = s_init or jnp.ones
        self.bias_shape = None

    def __call__(
        self,
        inputs: jnp.ndarray,
        multiplier: Union[float, jnp.ndarray] = None,
    ) -> jnp.ndarray:
      """Adds bias to ``inputs`` and optionally multiplies by ``multiplier``.
      Args:
        inputs: A Tensor of size ``[batch_size, input_size1, ...]``.
        multiplier: A scalar or Tensor which the bias term is multiplied by before
          adding it to ``inputs``. Anything which works in the expression ``bias *
          multiplier`` is acceptable here. This may be useful if you want to add a
          bias in one place and subtract the same bias in another place via
          ``multiplier=-1``.
      Returns:
        A Tensor of size ``[batch_size, input_size1, ...]``.
      """
      utils.assert_minimum_rank(inputs, 2)
      if self.output_size is not None and self.output_size != inputs.shape[1:]:
        raise ValueError(
            f"Input shape must be {(-1,) + self.output_size} not {inputs.shape}")

      self.bias_shape = calculate_bias_shape(inputs.shape, self.scale_dims)
      self.input_size = inputs.shape[1:]

      s = hk.get_parameter("s", self.bias_shape, inputs.dtype, init=self.s_init)
      s = jnp.broadcast_to(s, inputs.shape)

      return inputs * s


def calculate_bias_shape(input_shape: Sequence[int], bias_dims: Sequence[int]):
  """Calculate `bias_shape` based on the `input_shape` and `bias_dims`.
  Args:
    input_shape: Shape of the input being passed into the module. The leading
      dimension is the mini-batch size.
    bias_dims: The dimensions that bias should be applied over. The remaining
      dimensions will be broadcast over.
  Returns:
    bias_shape: Tuple corresponding to the shape of bias Variable to create.
  Raises:
    ValueError: If the user attempts to add bias over the mini-batch dimension,
        e.g. `bias_dims=[0]`.
  """
  input_rank = len(input_shape)
  if bias_dims is None:
    # If None, default is to use all dimensions.
    return input_shape[1:]

  elif not bias_dims:
    # If empty list, use a scalar bias.
    return ()

  else:
    # Otherwise, calculate bias_shape from bias_dims.
    bias_shape = [1] * input_rank
    # Populate bias dimensions.
    for dim in bias_dims:
      if dim < 0:
        dim %= input_rank

      if dim == 0:
        raise ValueError("Cannot apply bias across the minibatch dimension.")
      elif dim >= input_rank:
        raise ValueError(
            "Dimension %d (bias_dims=%r) out of range for input of rank %r." %
            (dim, tuple(bias_dims), input_rank))

      bias_shape[dim] = input_shape[dim]
    # Strip leading unit dimensions.
    start = input_rank
    for dim in range(1, input_rank):
      if bias_shape[dim] != 1:
        start = dim
        break
    return tuple(bias_shape[start:])  # Do not apply across minibatch dimension.

class FixupBlock(hk.Module):
  """Fixup ResNet block with optional bottleneck. https://openreview.net/pdf?id=H1gsz30cKX
  """

  def __init__(
      self,
      l: int,  # number of residual connections
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, float],
      bottleneck: bool,
      shortcut_last: Optional[bool] = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.shortcut_last = shortcut_last

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

    bias_0 = hk.Bias(bias_dims=[], name="b_0")
    channel_div = 4 if bottleneck else 1
    grp_layers_n = 3 if bottleneck else 2
    #  init_std = np.sqrt(2 / ((channels // channel_div) * 3 * 3)) * l ** (-0.5)
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        #  w_init=hk.initializers.TruncatedNormal(stddev=init_std),
        w_init=hk.initializers.VarianceScaling(2.0 * l**(-2/(2*grp_layers_n - 2)), "fan_in",  "truncated_normal"),
        padding="SAME",
        name="conv_0")

    self.bias_relu = hk.Bias(bias_dims=[], name="b_relu")

    bias_1 = hk.Bias(bias_dims=[], name="b_1")
    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        w_init=hk.initializers.Constant(0),
        padding="SAME",
        name="conv_1")

    layers = ((bias_0, conv_0,), (bias_1, conv_1,))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      layers = layers + ((conv_2,),)

    self.layers = layers
    self.scale = Scale(scale_dims=[], name="scale")
    self.bias_end = hk.Bias(bias_dims=[], name="b_end")

  def __call__(self, inputs):
      out = shortcut = inputs

      if self.use_projection:
        shortcut = self.proj_conv(shortcut)

      for i, (bias_i, conv_i,) in enumerate(self.layers):
          out = bias_i(out)
          out = conv_i(out)
          if i < len(self.layers) - 1:  # Don't apply relu on last layer
              out = self.bias_relu(out)
              out = jax.nn.relu(out)

      out = self.scale(out)
      out = self.bias_end(out)
      if not self.shortcut_last:
          return jax.nn.relu(out + shortcut)
      else:
          return jax.nn.relu(out) + shortcut

class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      l: int,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, float],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = FixupBlock

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(l=l,
                    channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
  """ResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
      200: {
          "blocks_per_group": (3, 24, 36, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
  }

  BlockGroup = BlockGroup  # pylint: disable=invalid-name

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      num_classes: int,
      bn_config: Optional[Mapping[str, float]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
  ):
    """Constructs a ResNet model.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    #  bn_config = dict(bn_config or {})
    #  bn_config.setdefault("decay_rate", 0.9)
    #  bn_config.setdefault("eps", 1e-5)
    #  bn_config.setdefault("create_scale", True)
    #  bn_config.setdefault("create_offset", True)

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", jnp.zeros)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    self.initial_conv = hk.Conv2D(
        output_channels=64,
        #  kernel_shape=7,
        #  stride=2,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding="SAME",
        name="initial_conv")

    #  if not self.resnet_v2:
      #  self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
                                            #  **bn_config)

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          BlockGroup(l=np.sum(blocks_per_group),
                     channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     name="block_group_%d" % (i)))

    #  if self.resnet_v2:
      #  self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs, is_training=False):
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = jax.nn.relu(out)

    #  out = hk.max_pool(out,
                      #  window_shape=(1, 3, 3, 1),
                      #  strides=(1, 2, 2, 1),
                      #  padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out)

    if self.resnet_v2:
      out = jax.nn.relu(out)
    out = jnp.mean(out, axis=[1, 2])
    return self.logits(out)

class FixupConvBlock(hk.Module):
  """Fixup Conv block with optional bottleneck. https://openreview.net/pdf?id=H1gsz30cKX
  """

  def __init__(
      self,
      l: int,  # number of residual connections
      in_channels: int,
      out_channels: int,
      stride: Union[int, Sequence[int]],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.bias_in = hk.Bias(bias_dims=[], name="b_in")

    #  grp_layers_n = 2

    self.conv = hk.Conv2D(
        output_channels=out_channels,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        #  w_init=hk.initializers.VarianceScaling(2.0 * l**(-2/(2*grp_layers_n - 2)), "fan_in",  "truncated_normal"),
        padding="SAME",
        name="conv")

    #  self.bias_relu = hk.Bias(bias_dims=[], name="b_relu")

    self.scale = Scale(scale_dims=[], name="scale")
    self.bias_end = hk.Bias(bias_dims=[], name="b_end")

  def __call__(self, inputs):
      out = inputs
      out = self.bias_in(out)
      out = self.conv(out)


      out = self.scale(out)
      out = self.bias_end(out)
      out = jax.nn.relu(out)
      out = hk.max_pool(out,
                        window_shape=(1, 2, 2, 1),
                        strides=(1, 2, 2, 1),
                        padding="SAME")
      return out

class ResNet9(hk.Module):
  """ResNet model."""
  def __init__(
      self,
      num_classes: int,
      name: Optional[str] = None,
  ):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      name: Name of the module.
    """
    super().__init__(name=name)

    logits_config = dict({})
    logits_config.setdefault("w_init", jnp.zeros)
    logits_config.setdefault("name", "logits")

    self.initial_conv = hk.Conv2D(
        output_channels=64,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding="SAME",
        name="initial_conv")

    self.layer1 = hk.Sequential([
        FixupConvBlock(l=2, in_channels=64, out_channels=128, stride=1, name="conv_block_1",),
        FixupBlock(l=2,
                  channels=128,
                  stride=1,
                  bottleneck=False,
                  bn_config=None,
                  use_projection=False,
                  shortcut_last=True,
                  name="block_1"),
        ])

    self.layer2 = FixupConvBlock(l=2, in_channels=128, out_channels=256, stride=1, name="conv_block_2",)

    self.layer3 = hk.Sequential([
        FixupConvBlock(l=2, in_channels=256, out_channels=512, stride=1, name="conv_block_3",),
        FixupBlock(l=2,
                  channels=512,
                  stride=1,
                  bottleneck=False,
                  bn_config=None,
                  use_projection=False,
                  shortcut_last=True,
                  name="block_2"),
        ])

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs, is_training=False):
      out = inputs
      out = self.initial_conv(out)
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)

      out = hk.max_pool(out,
                        window_shape=(1, 4, 4, 1),
                        strides=(1, 4, 4, 1),
                        padding="SAME")

      out = jnp.reshape(out, [out.shape[0], -1])
      out = self.logits(out)
      out = 0.125 * out

      return out

class ResNet18(ResNet):
  """ResNet18."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[18])


class ResNet34(ResNet):
  """ResNet34."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[34])


class ResNet50(ResNet):
  """ResNet50."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[50])


class ResNet101(ResNet):
  """ResNet101."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[101])


class ResNet152(ResNet):
  """ResNet152."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[152])


class ResNet200(ResNet):
  """ResNet200."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               logits_config: Optional[Mapping[str, Any]] = None,
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[200])
