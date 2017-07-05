import tensorflow as tf
import random
import numpy as np
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# tf.set_random_seed(777)

xy = np.loadtxt('test_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[0:3]
y_data = xy[3:]


# input place holders
X = tf.placeholder(tf.float32, [None, None])
Y = tf.placeholder(tf.float32, [None, None])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([None, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 3]))
b3 = tf.Variable(tf.random_normal([3]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define cost/loss & optimizer
learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range (2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

        # if step % 200 == 0:
        #     print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            # feed = {X:x_data, Y:y_data}
            # print ('{:8.6} {:8.6}'.format(sess.run(cost, feed_dict=feed)), sess.run(W))