import jax
import jax.numpy as jnp
import haiku as hk


class Classifier(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = self.config['num_classes']
        self.dropout_rate = self.config['dropout_rate']

        if self.config['activation'] == "relu":
            self.activation = jax.nn.relu
        elif self.config['activation'] == "leaky_relu":
            self.activation = lambda x: jax.nn.leaky_relu(x, negative_slope=self.config['negative_slope'])
        elif self.config['activation'] == "tanh":
            self.activation = jnp.tanh
        elif self.config['activation'] == "elu":
            self.activation = lambda x: jax.nn.elu(x, alpha=self.config['elu_alpha'])
        else:
            raise NotImplementedError


class CNNSmall(Classifier):
    def __call__(self, x, is_training=False):
        batch_size = x.shape[0]

        print_shapes=False

        # Conv layer 1
        if print_shapes: print(x.shape)
        x = hk.Conv2D(16, kernel_shape=8, stride=2, padding="VALID")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=1, padding="VALID")(x)
        if print_shapes: print(x.shape)

        # Conv layer 2
        x = hk.Conv2D(32, kernel_shape=4, stride=2)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=1, padding="VALID")(x)
        if print_shapes: print(x.shape)

        x = x.reshape(batch_size, -1)
        if print_shapes: print(x.shape)
        #  x = x.reshape(-1, 32 * 4 * 4)

        # Fully connected layer 1
        x = hk.Linear(32)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)

        # Fully connected layer 2
        x = hk.Linear(self.num_classes)(x)
        if print_shapes: print(x.shape)
        #  exit()

        return x


class CNNMed(Classifier):
    def __call__(self, x, is_training=False):
        batch_size = x.shape[0]

        print_shapes=False

        # Conv layer 1
        if print_shapes: print(x.shape)
        x = hk.Conv2D(32, kernel_shape=3, stride=1, padding="VALID")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if print_shapes: print(x.shape)

        # Conv layer 2
        x = hk.Conv2D(64, kernel_shape=3, stride=1)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if print_shapes: print(x.shape)

        # Conv layer 3
        x = hk.Conv2D(128, kernel_shape=3, stride=1)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if print_shapes: print(x.shape)

        # Conv layer 4
        x = hk.Conv2D(256, kernel_shape=3, stride=1)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)

        # Conv layer 5
        x = hk.Conv2D(self.num_classes, kernel_shape=3, stride=1)(x)
        if print_shapes: print(x.shape)

        x = hk.AvgPool(window_shape=x.shape[-2], strides=x.shape[-2], padding="VALID",
                channel_axis=-1)(x)
        #  x = hk.MaxPool(window_shape=x.shape[-2], strides=1, padding="VALID")(x)
        if print_shapes: print(x.shape)
        x = x.reshape(batch_size, -1)
        if print_shapes: print(x.shape)
        #  exit()

        return x


class VGG16(Classifier):
    @staticmethod
    def maybe_hk_dropout(rate, value):
        key = hk.maybe_next_rng_key()
        if key is not None:
            value = hk.dropout(key, rate, value)
        return value

    def __call__(self, x, is_training=False):
        batch_size = x.shape[0]

        print_shapes = False

        # Conv Block 1
        if print_shapes: print(x.shape)
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        if print_shapes: print(x.shape)

        # Conv Block 2
        if print_shapes: print(x.shape)
        x = hk.Conv2D(128, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(128, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        if print_shapes: print(x.shape)

        # Conv Block 3
        if print_shapes: print(x.shape)
        x = hk.Conv2D(256, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(256, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(256, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        if print_shapes: print(x.shape)

        # Conv Block 4
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        if print_shapes: print(x.shape)

        # Conv Block 5
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        x = hk.Conv2D(512, kernel_shape=3, stride=1, padding="SAME")(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        x = hk.MaxPool(window_shape=2, strides=2, padding="SAME")(x)
        if print_shapes: print(x.shape)

        # Flatten
        x = x.reshape(batch_size, -1)
        if print_shapes: print(x.shape)

        # Fully connected layer 1
        x = hk.Linear(4096)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        if self.dropout_rate is not None:
            x = self.maybe_hk_dropout(self.dropout_rate, x)
            if print_shapes: print(x.shape)

        # Fully connected layer 2
        x = hk.Linear(4096)(x)
        if print_shapes: print(x.shape)
        x = self.activation(x)
        if print_shapes: print(x.shape)
        if self.dropout_rate is not None:
            x = self.maybe_hk_dropout(self.dropout_rate, x)
            if print_shapes: print(x.shape)

        # Fully connected layer 3
        x = hk.Linear(self.num_classes)(x)
        if print_shapes: print(x.shape)

        return x