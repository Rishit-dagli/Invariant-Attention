import numpy as np
import tensorflow as tf
from einops import repeat

from invariant_attention.invariant_attention import InvariantPointAttention


class InvariantPointAttentionTest(tf.test.TestCase):
    def setUp(self):
        super(InvariantPointAttentionTest, self).setUp()

        self.attn = InvariantPointAttention(
            dim=64,
            heads=8,
            scalar_key_dim=16,
            scalar_value_dim=16,
            point_key_dim=4,
            point_value_dim=4,
        )

        self.single_repr = tf.random.normal((1, 256, 64))
        self.pairwise_repr = tf.random.normal((1, 256, 256, 64))
        self.mask = tf.ones((1, 256), dtype=tf.bool)
        self.rotations = repeat(tf.eye(3), "... -> b n ...", b=1, n=256)
        self.translations = tf.zeros((1, 256, 3))

    def test_shape_and_rank(self):
        outputs = self.attn(
            self.single_repr,
            self.pairwise_repr,
            rotations=self.rotations,
            translations=self.translations,
            mask=self.mask,
        )

        self.assertEqual(tf.rank(outputs), 3)
        self.assertShapeEqual(np.zeros((1, 256, 64)), outputs)


if __name__ == "__main__":
    tf.test.main()
