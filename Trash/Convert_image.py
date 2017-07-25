# Convert image between FastGfile and cv2
'''
import tensorflow as tf
import cv2
import numpy as np

#Loading the file
img2 = cv2.imread('/Users/User/PycharmProjects/network/ML_Camp/sliding_window/images/1.jpg')
# print(img2)
print('---')

#Format for the Mul:0 Tensor
# img2= cv2.resize(img2,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
#Numpy array
np_image_data = np.asarray(img2)
# print(np_image_data)
print('---')

#maybe insert float convertion here - see edit remark!
np_final = np.expand_dims(np_image_data, axis=0)
# print(np_final)

#now feeding it into the session:
#[... initialization of session and loading of graph etc]
# predictions = sess.run(softmax_tensor,
#                            {'Mul:0': np_final})
#fin!

print('===================================')
img2 = cv2.imread('/Users/User/PycharmProjects/network/ML_Camp/sliding_window/images/1.jpg') # 이미지를 불러옴

# 이미지를 배열로 불러옴(정수 배열) (inception v3 에서 분류하기 위해 이미지를 불러와서 윈도우로 나눔)
np_image_data = np.asarray(img2)
print(img2)
print('---')

# 바이트 스트링으로 변환된 이미지를 다시 정수 배열로 변환( 윈도우 작업을 계속하기 위해 다시 정수배열로 변환)
tobyte = img2.tobytes()
print(tobyte)
print('---')

# 바이트 스트링으로 변환된 이미지를 다시 정수 배열로 변환( 윈도우 작업을 계속하기 위해 다시 정수배열로 변환)
re_to_np = np.fromstring(tobyte, np.uint8).reshape(2207, 3, 3)
print(re_to_np)

print(re_to_np.shape)


'''



