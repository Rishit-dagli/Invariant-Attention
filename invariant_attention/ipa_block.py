from .invariant_attention import InvariantAttention
import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult = 1., num_layers = 2, activation = tf.keras.layers.ReLU, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.mult = mult
        self.num_layers = num_layers
        self.activation = activation
        dim_hidden = dim * mult

        self.layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            is_last  = ind == (num_layers - 1)
            if is_first:
                dim_in = dim
            else:
                dim_in = dim_hidden
            if is_last:
                dim_out = dim
            else:
                dim_out = dim_hidden

            self.layers.append(tf.keras.layers.Dense(dim_out, input_shape=(dim_in,)))

            if is_last:
                continue

            self.layers.append(activation())
        self.layers = tf.keras.Sequential(self.layers)
    
    def call(self, inputs):
        return self.layers(inputs)