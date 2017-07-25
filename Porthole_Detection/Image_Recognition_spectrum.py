import os, sys

import tensorflow as tf
import cv2

from scipy.misc import toimage  # 각 윈도우를 확인하는 뻘짓용
import imutils
import argparse
import time
import numpy as np
from PIL import Image
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# change this as you see fit
# image_path = sys.argv[1] # 콘솔창에서 실행이 안되므로 주석처리 실행 안되는 이유를 모르겠다 ㅠㅠ

# 이미지를 byte mode 로 불러들임
# image_data = tf.gfile.FastGFile('porthole_test.jpg', 'rb').read() # 수정됨, 경로 지정,

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/tmp/output_labels.txt")] # 수정됨, 경로지정 해야 에러안남 >> 위에껀 왜 필요한지 모르겠다.


image = cv2.imread('all(spectrum).jpg')


# 슬라이딩 윈도우를 사용하기 위한 함수
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Unpersists graph from file
with tf.gfile.FastGFile("/tmp/output_graph.pb", 'rb') as f: # 수정됨, 경로지정
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')



    # 여기다가 윈도우 슬라이드 코드를 넣고 슬라이드로 잘라낸 이미지를  아래 prediction 에 넣은 후 포트홀 일 경우에만 출력하는 방식으로 구현해보자
    # image_data 에 이미지 파일의 직접적인 경로 대신 슬라이드로 잘라낸 이미지를 반복적으로 넣는 방식으로 해결 가능할 것 같다.
    # 슬라이드가 주행기록을 모두 훑을 때 까지 prediction은 반복되어야 한다.
    # 시간 오래걸리겠다

    # 현재 문제점
    # 2. GPS 데이터를 어떻게 기록하고 포트홀일 경우 어떻게 출력하거나 지도에 표시할 것 인가?


    # 해결된 문제
    # 1. 이 파일에 윈도우를 어떻게 구현 할 것인가? >> 이건 끝난듯 히히 이제 합치기만 하면 된다. >> JPEG 파일이 아니라고 에러나는거랑 윈도우를 다시 배열로 바꿀 때 길이가 바뀌는걸 해결해야함 >> 해결


############################################################

    (winW, winH) = (6, len(image))  # 윈도우의 크기를 정한다, 만약 Input 데이터의 사이즈가 바뀌면 손댈 부분은 여기뿐
                            # 한반도 각 꼭짓점들의 차이는 위도 경도 둘다 약 3 정도 이다. 이정도면 GPS 데이터를 넣어도 상관 없나?


    # 윈도우가 이미지를 따라 스텝만큼 슬라이딩 한다
    for (x, y, window) in sliding_window(image, stepSize=2, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # print(x, y + 300) # 만약 이미지데이터에 GPS 정보까지 넣는다고 하면 포트홀이 감지되었을 때 Y번째에 있는 GPS 정보를 뽑아내면 될것
                    # 이건 슬라이딩 윈도우의 위치를 나타낸다(맨 윗 부분의 좌표 출력)

        # print(window) # 윈도우는 계속 이동하므로 윈도우 배열은 계속 바뀌고 있다.

        tmp = Image.fromarray(window) # tmp 변수에 window 의 정보를 넣고
        tmp.save('window.jpg') # window.jpg 파일로 저장
        tmp.close()
        # time.sleep(3.0)

        # image_data = tmp.tobytes() # 복제된 윈도우를 바이트 스트링으로 변환

        # 아마도 아래 예측 함수로 넘기기 전에 jpeg로 다시 인코딩 해줘야 하는 것 같다. window를 jpeg로 인코딩 후 다시 바이트스트링으로 바꿔야 할듯

        # 만들어진 window 를 classification 하기위해 원래 동작 방식 처럼 바이트 스트링으로 변환하여 예측 함수로 넘김  근데 윈도우 이미지가 계속 같은 값으로 나온다
        image_data = tf.gfile.FastGFile('window.jpg', 'rb').read()  # 이 부분을 해결하면 될것같긴 한데
        # print(image_data)

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # 슬라이딩 윈도우를 위해 cv2 파일로 불러온 이미지를 윈도우로 쪼갠 후 그 윈도우를 GFile(바이트 모드)로 변환해서 image_data 로 리턴해야 한다.

        # 그냥 윈도우를 그리는 코드
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        # time.sleep(0.025)

    ##############################################################

    # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k: # 각 라벨마다 node_id 가 정해져 있어서 스코어(확률)이 가장 높은 라벨부터 출력이 된다.
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if top_k[0] == 2: # 해당 라벨만 출력하게 함 ( 2가 포트홀 )
            # print(node_id)  # 각 라벨들의 node_id 출력
                print('%s (score = %.5f)' % (human_string, score))

        print("--------------------------------------------")

'''
        for node_id in top_k: # 각 라벨마다 node_id 가 정해져 있어서 스코어(확률)이 가장 높은 라벨부터 출력이 된다.
            human_string = label_lines[node_id]
            score = predictions[0][node_id]

            print('%s (score = %.5f)' % (human_string, score))

        print("--------------------------------------------")
'''