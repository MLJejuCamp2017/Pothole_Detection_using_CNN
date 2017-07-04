import tensorflow as tf
import numpy as np

xy = np.loadtxt('test_data.csv', unpack=True, delimiter=',', dtype='float32')
x_data = np.transpose(xy[0:3])  # 한 행에 Z X Y 가 들어감, 즉 test_data.csv 상태 그대로 한 행씩 읽어옴
y_data = np.transpose(xy[-1])  # 한 행에 Label 이 들어감, np에 의해 test_data.csv 가 전치되어 들어왔으므로 소프트맥스 연산을 위해 이렇게 가져온다

print(x_data)
print(y_data)

X = tf.placeholder("float32", shape = [None, None])  # [None, 3]  >> 행의 크기는 모르지만 열의 크기가 3인 행렬,  [None, None] 해도 무관
Y = tf.placeholder("float32", shape = [None, None])  # "float" 말고 tf.float32 라고 써도 됨

W = tf.Variable(tf.zeros([3, 3]))  # 모든 내용이 0인 3x3 행렬,  tf.zeros([binary_Classification 횟수, x_data의 크기])



# # weights & bias for nn layers
# W1 = tf.Variable(tf.random_normal([1, 20]))
# b1 = tf.Variable(tf.random_normal([20]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([20, 20]))
# b2 = tf.Variable(tf.random_normal([20]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
#
# W3 = tf.Variable(tf.random_normal([20, 1]))
# b3 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.matmul(L2, W3) + b3




# 행렬의 형태 >>  X,Y=[8, 3], W=[3, 3]
hypothesis = tf.nn.softmax(tf.matmul(X, W))  # 소프트맥스에선 W*x 가 아닌 X*W 이므로 x_data 와 y_data 를 transpose 시켜서 가져온 것이다.
                                               # 위에서 transpose 를 안하면 그대로 W*X 해도 상관 없지만 결과를 추출할 때 불편해진다

learning_rate = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1)) # reduction_indices 가 1이면 행 기준 합계를 적용한다.
# cross-entropy cost 함수의 TensorFlow 버전이다. log 함수를 호출하여 hypothesis 를 처리한다. hypothesis는 이미 softmax를 거쳤으므로 0과 1사이의 값만 가진다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})























# import tensorflow as tf
# import numpy as np
#
# xy = np.loadtxt('test_data.csv', unpack=True, delimiter=',', dtype='float32')
# x_data = np.transpose(xy[0:3])  # 한 행에 Z X Y 가 들어감, 즉 test_data.csv 상태 그대로 한 행씩 읽어옴
# y_data = np.transpose(xy[-1])  # 한 행에 Label 이 들어감, np에 의해 test_data.csv 가 전치되어 들어왔으므로 소프트맥스 연산을 위해 이렇게 가져온다
#
# print(x_data)
# print(y_data)
#
# X = tf.placeholder("float32", [None, 1])  # [None, 3]  >> 행의 크기는 모르지만 열의 크기가 3인 행렬,  [None, None] 해도 무관
# Y = tf.placeholder("float32", [None, 1])  # "float" 말고 tf.float32 라고 써도 됨
#
# # W = tf.Variable(tf.zeros([3, 3]))  # 모든 내용이 0인 3x3 행렬,  tf.zeros([binary_Classification 횟수, x_data의 크기])
#
#
#
# # weights & bias for nn layers
# W1 = tf.Variable(tf.random_normal([1, 20]))
# b1 = tf.Variable(tf.random_normal([20]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([20, 20]))
# b2 = tf.Variable(tf.random_normal([20]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
#
# W3 = tf.Variable(tf.random_normal([20, 1]))
# b3 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.matmul(L2, W3) + b3
#
#
#
#
# # 행렬의 형태 >>  X,Y=[8, 3], W=[3, 3]
# # hypothesis = tf.nn.softmax(tf.matmul(X, W))  # 소프트맥스에선 W*x 가 아닌 X*W 이므로 x_data 와 y_data 를 transpose 시켜서 가져온 것이다.
#                                               #  위에서 transpose 를 안하면 그대로 W*X 해도 상관 없지만 결과를 추출할 때 불편해진다
#
# # define cost/loss & optimizer
# learning_rate = 0.001
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # # initialize
# # sess = tf.Session()
# # sess.run(tf.global_variables_initializer())
#
#
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for step in range(2001):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         # if step % 200 == 0:
#             # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
#
#
#
#
#
#
#
#
#
#
# # learning_rate = 0.001
# # cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1)) # reduction_indices 가 1이면 행 기준 합계를 적용한다.
# # # cross-entropy cost 함수의 TensorFlow 버전이다. log 함수를 호출하여 hypothesis 를 처리한다. hypothesis는 이미 softmax를 거쳤으므로 0과 1사이의 값만 가진다.
# #
#
