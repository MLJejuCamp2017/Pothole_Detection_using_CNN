# from matplotlib import pyplot as plt
# from scipy.misc import imshow
# import numpy as np
# from scipy.misc import toimage
# xy = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/porthole/original/log.csv",unpack=True, delimiter=',', dtype=np.float32)
# plt.figure(figsize=(20,10))
# toimage(xy).show()
#
# plt.imshow(xy, interpolation='nearest')
# plt.savefig("log.jpg")
# plt.imsave("log.jpg")


# import numpy as np
# from matplotlib import pyplot as plt
# # x = numpy.random.rand(10, 10)*255
# x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/porthole/original/log.csv", delimiter=',')
#
# np.reshape(x, (35,[]))
#
# plt.imshow(x, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.savefig("txt.png")
# plt.show()

# 파일 하나만 바꿔줌

import numpy as np
from scipy.misc import toimage

x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',')
# toimage(x).show()
toimage(x).save('all.jpg')

'''

# 디렉토리 내의 파일들을 한번에 일괄 변환 와 개쩐다 난 노가다했는데 ㅅㅂ 진작에 할걸

import os
import numpy as np
from scipy.misc import toimage

path = "/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/asphalt/"
dirs = os.listdir(path)

def convert():
    for item in dirs:
        if os.path.isfile(path+item):
            print(path+item)
            x = np.loadtxt(path+item, delimiter=',')
            f, e = os.path.splitext(path+item)
            toimage(x).save(f + '.jpg')

convert()
'''