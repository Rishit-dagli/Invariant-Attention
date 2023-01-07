import tensorflow as tf

def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    """
    aw, ax, ay, az = tf.unstack(a, axis=-1)
    bw, bx, by, bz = tf.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return tf.stack([ow, ox, oy, oz], axis = -1)

