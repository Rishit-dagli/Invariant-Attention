import numpy as np
import tensorflow as tf
from einops import repeat

from invariant_attention.ipa_transformer import IPATransformer


class IPATransformerTest(tf.test.TestCase):
    def setUp(self):
        super(IPATransformerTest, self).setUp()

        seq = tf.random.normal((1, 256, 32))
        pairwise_repr = tf.random.normal((1, 256, 256, 32))
        mask = tf.ones((1, 256), dtype=tf.bool)
        translations = tf.zeros((1, 256, 3))

        model = IPATransformer(
            dim=32,
            depth=2,
            num_tokens=None,
            predict_points=False,
            detach_rotations=True,
        )

        self.outputs = model(
            single_repr=seq,
            translations=translations,
            quaternions=tf.random.normal((1, 256, 4)),
            pairwise_repr=pairwise_repr,
            mask=mask,
        )

    def test_shape_and_rank_1(self):
        self.assertEqual(tf.rank(self.outputs[0]), 3)
        self.assertShapeEqual(np.zeros((1, 256, 32)), self.outputs[0])

    def test_shape_and_rank_2(self):
        self.assertEqual(tf.rank(self.outputs[1]), 3)
        self.assertShapeEqual(np.zeros((1, 256, 3)), self.outputs[1])

    def test_shape_and_rank_3(self):
        self.assertEqual(tf.rank(self.outputs[2]), 3)
        self.assertShapeEqual(np.zeros((1, 256, 4)), self.outputs[2])


if __name__ == "__main__":
    tf.test.main()
