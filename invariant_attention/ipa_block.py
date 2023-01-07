import tensorflow as tf

from .invariant_attention import InvariantPointAttention


class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self, dim, mult=1.0, num_layers=2, activation=tf.keras.layers.ReLU, **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.mult = mult
        self.num_layers = num_layers
        self.activation = activation
        dim_hidden = dim * mult

        self.layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            is_last = ind == (num_layers - 1)
            if is_first:
                dim_in = dim
            else:
                dim_in = dim_hidden
            if is_last:
                dim_out = dim
            else:
                dim_out = dim_hidden
            self.layers.append(tf.keras.layers.Dense(dim_out))

            if is_last:
                continue

            self.layers.append(activation())
        self.layers = tf.keras.Sequential(self.layers)

    def call(self, inputs):
        return self.layers(inputs)


class IPABlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        ff_mult=1,
        ff_num_layers=3,
        post_norm=True,
        post_attn_dropout=0.0,
        post_ff_dropout=0.0,
        **kwargs
    ):
        super(IPABlock, self).__init__()
        self.post_norm = post_norm

        self.attn_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.attn = InvariantPointAttention(dim=dim, **kwargs)
        self.post_attn_dropout = tf.keras.layers.Dropout(post_attn_dropout)

        self.ff_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.ff = FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers)
        self.post_ff_dropout = tf.keras.layers.Dropout(post_ff_dropout)

    def call(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        x = self.ff_norm(x) if post_norm else x
        return x
