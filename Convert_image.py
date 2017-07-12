# Convert image between FastGfile and cv2

import tensorflow as tf
import cv2
import numpy as np

#Loading the file
img2 = cv2.imread('/Users/User/PycharmProjects/network/ML_Camp/sliding_window/images/1.jpg')
print(img2)
print('---')

#Format for the Mul:0 Tensor
# img2= cv2.resize(img2,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
#Numpy array
np_image_data = np.asarray(img2)
print(np_image_data)
print('---')

#maybe insert float convertion here - see edit remark!
np_final = np.expand_dims(np_image_data, axis=0)
print(np_final)

#now feeding it into the session:
#[... initialization of session and loading of graph etc]
# predictions = sess.run(softmax_tensor,
#                            {'Mul:0': np_final})
#fin!