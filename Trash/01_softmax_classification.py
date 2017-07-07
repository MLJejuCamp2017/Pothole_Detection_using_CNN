# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf

# # x0 x1 x2 y[A  B  C]
#   1  2  1    0  0  1
#   1  3  2    0  0  1
#   1  3  4    0  0  1
#   1  5  5    0  1  0
#   1  7  5    0  1  0
#   1  2  5    0  1  0
#   1  6  6    1  0  0
#   1  7  7    1  0  0

xy = np.loadtxt('test_data.csv', delimiter=',', unpack=True, dtype='float32') # train.txt 를 전치시켜서 가져온다. 각 행당 [x0] [x1] [2].. 이렇게 들어간다
x_data = np.transpose(xy[0:3])  # 한 행에 [x0, x1, x2] 가 들어감, 즉 train.txt 상태 그대로 한 행씩 읽어옴
y_data = np.transpose(xy[-1])   # 한 행에 [yA yB yC] 가 들어감, np에 의해 train.txt 가 전치되어 들어왔으므로 소프트맥스 연산을 위해 이렇게 가져온다

X = tf.placeholder("float", [None, 1])  # [None, 3]  >> 행의 크기는 모르지만 열의 크기가 3인 행렬,  [None, None] 해도 무관
Y = tf.placeholder("float", [None, 1])  # "float" 말고 tf.float32 라고 써도 됨

W = tf.Variable(tf.zeros([1, 1]))  # 모든 내용이 0인 3x3 행렬,  tf.zeros([binary_Classification 횟수, x_data의 크기])

# 행렬의 형태 >>  X,Y=[8, 3], W=[3, 3]
hypothesis = tf.nn.softmax(tf.matmul(X, W))  # 소프트맥스에선 W*x 가 아닌 X*W 이므로 x_data 와 y_data 를 transpose 시켜서 가져온 것이다.
                                              #  위에서 transpose 를 안하면 그대로 W*X 해도 상관 없지만 결과를 추출할 때 불편해진다

learning_rate = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1)) # reduction_indices 가 1이면 행 기준 합계를 적용한다.
# cross-entropy cost 함수의 TensorFlow 버전이다. log 함수를 호출하여 hypothesis 를 처리한다. hypothesis는 이미 softmax를 거쳤으므로 0과 1사이의 값만 가진다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        # if step % 200 == 0:
            # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print(y_data)
# from __future__ import print_function
#
# import numpy as np
# import tensorflow as tf
#
# class Softmax_classification:
#     def __init__(self):
#
#         self.xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
#         self.x_data = np.transpose(self.xy[0:3])  #transpose : 전치행렬
#         self.y_data = np.transpose(self.xy[3:])
#
#
#         self.X = tf.placeholder("float", [None, 3])
#         self.Y = tf.placeholder("float", [None, 3])
#
#         self.W = tf.Variable(tf.zeros([3, 3]))
#
#     def Hypothesis(self):
#         # matrix shape X=[8, 3], W=[3, 3]
#         self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W))
#
#         self.learning_rate = 0.001
#
#     def gradient_descent(self):
#         self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), reduction_indices=1))
#         self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
#
#
#
#     def initialize(self):
#         self.init = tf.global_variables_initializer()
#         self.sess = tf.Session()
#         self.sess.run(self.init)
#
#
# sc=Softmax_classification()
# sc.Hypothesis()
# sc.gradient_descent()
# sc.initialize()
#
#
# with tf.Session() as sess:
#     sc.sess.run(sc.init)
#
#     for step in range(2001):
#         sc.sess.run(sc.optimizer, feed_dict={sc.X: sc.x_data, sc.Y: sc.y_data})
#         if step % 20 == 0:
#             print(step, sc.sess.run(sc.cost, feed_dict={sc.X: sc.x_data, sc.Y: sc.y_data}), sc.sess.run(sc.W))
