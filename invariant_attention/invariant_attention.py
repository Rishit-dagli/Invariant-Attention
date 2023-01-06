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

        point_weight_init_value = tf.math.log(tf.math.exp(tf.ones((heads,))) - 1.)
        self.point_weights = tf.Variable(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = tf.keras.layers.Dense(point_key_dim * heads * 3, use_bias = False)
        self.to_point_k = tf.keras.layers.Dense(point_key_dim * heads * 3, use_bias = False)
        self.to_point_v = tf.keras.layers.Dense(point_value_dim * heads * 3, use_bias = False)

        if require_pairwise_repr:
            if pairwise_repr_dim is not None:
                pairwise_repr_dim = pairwise_repr_dim
            else:
                pairwise_repr_dim = dim
        else:
            pairwise_repr_dim = 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = tf.keras.Sequential([
                tf.keras.layers.Dense(heads),
                Rearrange('b ... h -> (b h) ...')
            ])
        
        self.to_out = tf.keras.layers.Dense(dim)
    
    def call(
        self,
        single_repr,
        pairwise_repr = None,
        *,
        rotations,
        translations,
        mask = None
    ):
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
