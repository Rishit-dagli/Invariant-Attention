import tensorflow as tf
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange


class InvariantPointAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        heads=8,
        scalar_key_dim=16,
        scalar_value_dim=16,
        point_key_dim=4,
        point_value_dim=4,
        pairwise_repr_dim=None,
        require_pairwise_repr=True,
        eps=1e-8,
        **kwargs
    ):
        super(InvariantPointAttention, self).__init__(**kwargs)

        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        num_attn_logits = 3 if require_pairwise_repr else 2

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = tf.keras.layers.Dense(scalar_key_dim * heads, use_bias=False)
        self.to_scalar_k = tf.keras.layers.Dense(scalar_key_dim * heads, use_bias=False)
        self.to_scalar_v = tf.keras.layers.Dense(
            scalar_value_dim * heads, use_bias=False
        )

        point_weight_init_value = tf.math.log(tf.math.exp(tf.ones((heads,))) - 1.0)
        self.point_weights = tf.Variable(point_weight_init_value)

        self.point_attn_logits_scale = (
            (num_attn_logits * point_key_dim) * (9 / 2)
        ) ** -0.5

        self.to_point_q = tf.keras.layers.Dense(
            point_key_dim * heads * 3, use_bias=False
        )
        self.to_point_k = tf.keras.layers.Dense(
            point_key_dim * heads * 3, use_bias=False
        )
        self.to_point_v = tf.keras.layers.Dense(
            point_value_dim * heads * 3, use_bias=False
        )

        if require_pairwise_repr:
            if pairwise_repr_dim is not None:
                pairwise_repr_dim = pairwise_repr_dim
            else:
                pairwise_repr_dim = dim
        else:
            pairwise_repr_dim = 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits**-0.5

            self.to_pairwise_attn_bias = tf.keras.Sequential(
                [tf.keras.layers.Dense(heads), Rearrange("b ... h -> (b h) ...")]
            )

        self.to_out = tf.keras.layers.Dense(dim)

    def call(
        self, single_repr, pairwise_repr=None, *, rotations, translations, mask=None
    ):
        x, b, h, eps, require_pairwise_repr = (
            single_repr,
            single_repr.shape[0],
            self.heads,
            self.eps,
            self.require_pairwise_repr,
        )
        assert not (
            require_pairwise_repr and pairwise_repr is None
        ), "pairwise representation cannot be empty if require_pairwise_repr is True"

        q_scalar, k_scalar, v_scalar = (
            self.to_scalar_q(x),
            self.to_scalar_k(x),
            self.to_scalar_v(x),
        )

        q_point, k_point, v_point = (
            self.to_point_q(x),
            self.to_point_k(x),
            self.to_point_v(x),
        )

        q_scalar, k_scalar, v_scalar = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h),
            (q_scalar, k_scalar, v_scalar),
        )
        q_point, k_point, v_point = map(
            lambda t: rearrange(t, "b n (h d c) -> (b h) n d c", h=h, c=3),
            (q_point, k_point, v_point),
        )

        rotations = repeat(rotations, "b n r1 r2 -> (b h) n r1 r2", h=h)
        translations = repeat(translations, "b n c -> (b h) n () c", h=h)

        q_point = (
            tf.einsum("b n d c, b n c r -> b n d r", q_point, rotations) + translations
        )
        k_point = (
            tf.einsum("b n d c, b n c r -> b n d r", k_point, rotations) + translations
        )
        v_point = (
            tf.einsum("b n d c, b n c r -> b n d r", v_point, rotations) + translations
        )

        attn_logits_scalar = (
            tf.einsum("b i d, b j d -> b i j", q_scalar, k_scalar)
            * self.scalar_attn_logits_scale
        )

        if require_pairwise_repr:
            attn_logits_pairwise = (
                self.to_pairwise_attn_bias(pairwise_repr)
                * self.pairwise_attn_logits_scale
            )

        point_qk_diff = rearrange(q_point, "b i d c -> b i () d c") - rearrange(
            k_point, "b j d c -> b () j d c"
        )
        point_dist = tf.reduce_sum((point_qk_diff**2), axis=(-1, -2))

        point_weights = tf.math.softplus(self.point_weights)
        point_weights = repeat(point_weights, "h -> (b h) () ()", b=b)

        attn_logits_points = -0.5 * (
            point_dist * point_weights * self.point_attn_logits_scale
        )

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            mask = rearrange(mask, "b i -> b i ()") * rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            mask = tf.cast(mask, dtype=tf.bool)
            mask_value = -tf.experimental.numpy.finfo(attn_logits.dtype).max
            attn_logits = tf.where(mask, mask_value, attn_logits)

        attn = tf.nn.softmax(attn_logits, axis=-1)

        results_scalar = tf.einsum("b i j, b j d -> b i d", attn, v_scalar)
        attn_with_heads = rearrange(attn, "(b h) i j -> b h i j", h=h)

        if require_pairwise_repr:
            results_pairwise = tf.einsum(
                "b h i j, b i j d -> b h i d", attn_with_heads, pairwise_repr
            )

        results_points = tf.einsum("b i j, b j d c -> b i d c", attn, v_point)

        rotation_rank = tf.rank(rotations)
        perm = list(range(rotation_rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        results_points = tf.einsum(
            "b n d c, b n c r -> b n d r",
            results_points - translations,
            tf.transpose(rotations, perm=perm),
        )
        results_points_norm = tf.math.sqrt(
            tf.reduce_sum(tf.math.square(results_points), axis=-1) + eps
        )

        results_scalar = rearrange(results_scalar, "(b h) n d -> b n (h d)", h=h)
        results_points = rearrange(results_points, "(b h) n d c -> b n (h d c)", h=h)
        results_points_norm = rearrange(
            results_points_norm, "(b h) n d -> b n (h d)", h=h
        )

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, "b h n d -> b n (h d)", h=h)
            results = (*results, results_pairwise)

        results = tf.concat(results, axis=-1)
        return self.to_out(results)
