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

def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    """
    condition = quaternions[..., 0:1] < 0
    x = -quaternions
    y = quaternions
    return tf.where(condition, x, y)

def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)