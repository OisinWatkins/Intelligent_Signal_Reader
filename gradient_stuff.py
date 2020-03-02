import tensorflow as tf

# def log1pexp(x):
#     return tf.log(1 + tf.exp(x))
#
# x = tf.constant(100.)
# y = log1pexp(x)
# dy = tf.gradients(y, x)
#
# with tf.Session() as sess:
#     print(sess.run(dy))


@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)

    def grad(dy):
        return dy * (1 - 1 / (1 + e))

    return tf.log(1 + e), grad


x = tf.constant(100.)
y = log1pexp(x)
dy = tf.gradients(y, x)

with tf.Session() as sess:
    print(sess.run(dy))
