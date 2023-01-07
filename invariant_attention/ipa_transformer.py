import tensorflow as tf

from .ipa_block import IPABlock
from .quaternion_ops import quaternion_multiply, quaternion_to_matrix


class IPATransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        depth,
        num_tokens=None,
        predict_points=False,
        detach_rotations=True,
        **kwargs
    ):

        super(IPATransformer, self).__init__(**kwargs)
        self.quaternion_to_matrix = quaternion_to_matrix
        self.quaternion_multiply = quaternion_multiply

        if num_tokens is not None:
            self.token_emb = tf.keras.layers.Embedding(num_tokens, dim)
        else:
            self.token_emb = None

        self.layers = []
        for _ in range(depth):
            ipa_block = IPABlock(dim=dim, **kwargs)
            linear_layer = tf.keras.layers.Dense(6)
            self.layers.append([ipa_block, linear_layer])

        self.detach_rotations = detach_rotations
        self.predict_points = predict_points

        if predict_points:
            self.to_points = tf.keras.layers.Dense(3)

    def call(
        self,
        single_repr,
        translations=None,
        quaternions=None,
        pairwise_repr=None,
        mask=None,
    ):

        x, quaternion_multiply, quaternion_to_matrix = (
            single_repr,
            self.quaternion_multiply,
            self.quaternion_to_matrix,
        )
        b = x.shape[0]
        n = x.shape[1]

        if self.token_emb is not None:
            x = self.token_emb(x)

        if quaternions is None:
            quaternions = tf.constant([1.0, 0.0, 0.0, 0.0])
            quaternions = repeat(quaternions, "d -> b n d", b=b, n=n)

        if translations is None:
            translations = tf.zeros((b, n, 3))

        for block, to_update in self.layers:
            rotations = quaternion_to_matrix(quaternions)

            if self.detach_rotations:
                rotations = tf.stop_gradient(rotations)

            x = block(
                x,
                pairwise_repr=pairwise_repr,
                rotations=rotations,
                translations=translations,
            )

            quaternion_update, translation_update = tf.split(
                to_update(x), num_or_size_splits=2, axis=-1
            )
            quaternion_update_rank = tf.rank(quaternion_update)
            padding = []
            for _ in range(quaternion_update_rank - 1):
                padding.append([0, 0])
            padding.append([1, 0])
            quaternion_update = tf.pad(quaternion_update, padding, constant_values=1.0)
            quaternion_update = quaternion_update / tf.norm(
                quaternion_update, ord="euclidean", axis=-1, keepdims=True
            )
            quaternions = quaternion_multiply(quaternions, quaternion_update)
            translations = translations + tf.einsum(
                "b n c, b n c r -> b n r", translation_update, rotations
            )

        if not self.predict_points:
            return x, translations, quaternions

        points_local = self.to_points(x)
        rotations = quaternion_to_matrix(quaternions)
        points_global = (
            tf.einsum("b n c, b n c d -> b n d", points_local, rotations) + translations
        )
        return points_global
