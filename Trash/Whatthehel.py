from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

xy = np.loadtxt('gyro_acc.csv', unpack=True, delimiter=',', dtype='float32')
x_data = np.transpose(xy[0:6])  # 한 행에 Z X Y 가 들어감, 즉 test_data.csv 상태 그대로 한 행씩 읽어옴
y_data = np.transpose(xy[6:])  # 한 행에 Label 이 들어감, np에 의해 test_data.csv 가 전치되어 들어왔으므로 소프트맥스 연산을 위해 이렇게 가져온다

######테스트용 데이터   1
test = np.loadtxt('gyro_acc_test.csv', unpack=True, delimiter=',', dtype='float32')
train_data = np.transpose(test[0:6])
print(train_data.shape)
#########   1


X = tf.placeholder("float", shape=[None, 6])
Y = tf.placeholder("float", shape=[None, 4])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W1 = weight_variable([None, 32])
b1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(Y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(200):
   if i%10 == 0:
     train_accuracy = sess.run(accuracy, feed_dict={X:x_data, Y: y_data, keep_prob: 1.0})
     print("step %d, training accuracy %g"%(i, train_accuracy))
   sess.run(train_step, feed_dict={X: x_data, Y: y_data, keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
       X: x_data, Y: y_data, keep_prob: 1.0}))