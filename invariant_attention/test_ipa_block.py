import numpy as np
import tensorflow as tf
from einops import repeat

from invariant_attention.ipa_block import IPABlock


class IPABlockTest(tf.test.TestCase):
    def setUp(self):
        super(IPABlockTest, self).setUp()

        self.block = IPABlock(
            dim=64,
            heads=8,
            scalar_key_dim=16,
            scalar_value_dim=16,
            point_key_dim=4,
            point_value_dim=4,
        )

        self.seq = tf.random.normal((1, 256, 64))
        self.pairwise_repr = tf.random.normal((1, 256, 256, 64))
        self.mask = tf.ones((1, 256), dtype=tf.bool)
        self.rotations = repeat(tf.eye(3), "... -> b n ...", b=1, n=256)
        self.translations = tf.zeros((1, 256, 3))

        outputs = self.block(
            self.seq,
            pairwise_repr=self.pairwise_repr,
            rotations=self.rotations,
            translations=self.translations,
            mask=self.mask,
        )

        updates = tf.keras.layers.Dense(6)(outputs)
        self.quaternion_update, self.translation_update = tf.split(
            updates, num_or_size_splits=2, axis=-1
        )

    def test_shape_and_rank_translation_update(self):
        self.assertEqual(tf.rank(self.translation_update), 3)
        self.assertShapeEqual(np.zeros((1, 256, 3)), self.translation_update)

    def test_shape_and_rank_quaternion_update(self):
        self.assertEqual(tf.rank(self.quaternion_update), 3)
        self.assertShapeEqual(np.zeros((1, 256, 3)), self.quaternion_update)


if __name__ == "__main__":
    tf.test.main()
