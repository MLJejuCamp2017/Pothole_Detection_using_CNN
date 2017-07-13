import os, sys

import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# change this as you see fit
# image_path = sys.argv[1] # 콘솔창에서 실행이 안되므로 주석처리 실행 안되는 이유를 모르겠다 ㅠㅠ

# 이미지를 byte mode 로 불러들임
image_data = tf.gfile.FastGFile('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/porthole_test.png', 'rb').read() # 수정됨, 경로 지정,

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/tmp/output_labels.txt")] # 수정됨, 경로지정 해야 에러안남 >> 위에껀 왜 필요한지 모르겠다.


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
    # 1. 이 파일에 윈도우를 어떻게 구현 할 것인가? >> 이건 끝난듯 히히 이제 합치기만 하면 된다.
    # 2. GPS 데이터를 어떻게 기록하고 포트홀일 경우 어떻게 출력하거나 지도에 표시할 것 인가? 이미지파일에 모두 담기엔 조금 그런데;;
    # 3. SensorStream IMU+GPS 어플리케이션을 현재 안드로이드 버전에서 GPS가 작동하게 만들어야 한다. (마그네틱 센서와 GPS 센서 문제)
    # 4. 3번 문제를 해결하기 위해 일단 어플리케이션 코드를 분석해야하는데 김도현 교수님께 한번 여쭤보자


############################################################

    image = cv2.imread('speedbump_test.jpg')
    (winW, winH) = (3, 128)  # 윈도우의 크기를 정한다

    # 윈도우가 이미지를 따라 스텝만큼 슬라이딩 한다
    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # 슬라이딩 윈도우를 위해 cv2 파일로 불러온 이미지를 윈도우로 쪼갠 후 그 윈도우를 GFile(바이트 모드)로 변환해서 image_data 로 리턴해야 한다.
        image_data = tf.gfile.FastGFile(window, 'rb').read()  # 이 부분을 해결하면 될것같긴 한데


##############################################################

    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

        # 가장 가능성이 큰 라벨부터 출력(분류) 하므로 모델이 무엇으로 판단했는지 확인
        # if label_lines[0] == "speedbump":
        #     print("gg")
