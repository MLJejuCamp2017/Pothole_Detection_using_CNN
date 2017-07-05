import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('Acc_data.csv', unpack=True, delimiter=',', dtype='float32')
x_data = np.transpose(xy[0:3])  # 한 행에 Z X Y 가 들어감, 즉 test_data.csv 상태 그대로 한 행씩 읽어옴
y_data = np.transpose(xy[3:])  # 한 행에 Label 이 들어감, np에 의해 test_data.csv 가 전치되어 들어왔으므로 소프트맥스 연산을 위해 이렇게 가져온다

#테스트용 데이터
test = np.loadtxt('test.csv', unpack=True, delimiter=',', dtype='float32')
train_data = np.transpose(test[0:3])

print(train_data.shape)
print('xy.shape :', xy.shape)
print('x_data shape :', x_data.shape)
print('y_data shape :', y_data.shape)

print(len(x_data))
print(len(y_data))

########################

X = tf.placeholder(tf.float32, [None, None])
Y = tf.placeholder(tf.float32, [None, None])

# W = tf.Variable(tf.zeros([3, 3]))


W1 = tf.Variable(tf.random_normal([3, 10]))
b1 = tf.Variable(tf.random_normal([10]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 3]))
b2 = tf.Variable(tf.random_normal([3]))


#softmax 알고리듬 적용
hypothesis = tf.nn.softmax(tf.matmul(L1, W2))

# cross-entropy 함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

learning_rate = 0.00003
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range (20000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 200 == 0:
            print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            # feed = {X:x_data, Y:y_data}
            # print ('{:8.6} {:8.6}'.format(sess.run(cost, feed_dict=feed)), sess.run(W))

    print(sess.run(hypothesis, feed_dict={X: [[-1.1707647,2.7988217,8.635885],
                                              [-0.7386112,2.9149406,8.511387],
                                          [-0.17597382,3.0226796,8.456321]]}))