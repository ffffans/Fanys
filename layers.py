import tensorflow as tf


def weight_variable(shape, stddev=0.1, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def conv2d(x, w, b, drop_out):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, drop_out)


def deconv2d(x, w, stride=2):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] // stride])
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding="SAME")


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding="SAME")


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")



