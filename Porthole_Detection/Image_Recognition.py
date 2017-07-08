import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
# image_path = sys.argv[1] # 콘솔창에서 실행이 안되므로 주석처리 실행 안되는 이유를 모르겠다 ㅠㅠ

# Read in the image_data
image_data = tf.gfile.FastGFile('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/porthole_test.jpg', 'rb').read() # 수정됨, 경로 지정

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/tmp/output_labels.txt")] # 수정됨, 경로지정 해야 에러안남 >> 위에껀 왜 필요한지 모르겠다.

# Unpersists graph from file
with tf.gfile.FastGFile("/tmp/output_graph.pb", 'rb') as f: # 수정됨, 경로지정
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))