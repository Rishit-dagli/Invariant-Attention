import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from einops import rearrange, repeat

class InvariantPointAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4,
        pairwise_repr_dim = None,
        require_pairwise_repr = True,
        eps = 1e-8,
        **kwargs
    ):
        super(InvariantPointAttention, self).__init__(**kwargs)

        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        num_attn_logits = 3 if require_pairwise_repr else 2

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = tf.keras.layers.Dense(scalar_key_dim * heads, use_bias = False)
        self.to_scalar_k = tf.keras.layers.Dense(scalar_key_dim * heads, use_bias = False)
        self.to_scalar_v = tf.keras.layers.Dense(scalar_value_dim * heads, use_bias = False)

