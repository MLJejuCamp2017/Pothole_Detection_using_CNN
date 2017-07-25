# 파일 하나만 바꿔줌
'''
import numpy as np
from scipy.misc import toimage

x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',')
# toimage(x).show()
toimage(x).save('all(grayscale).jpg')

'''
'''
# 디렉토리 내의 파일들을 한번에 일괄 변환 와 개쩐다 난 노가다했는데 ㅅㅂ 진작에 할걸

import os
import numpy as np
from scipy.misc import toimage

path = "/Users/User/OneDrive/센서로그/자전거/포트홀/csv/다듬다듬/"
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

# 스펙트럼 이미지로 일괄 변환
# '''
import matplotlib.pyplot as plt
import stft
import os
import numpy as np

path = "/Users/User/OneDrive/센서로그/자전거/포트홀/csv/다듬다듬/"
dirs = os.listdir(path)

def convert():
    for item in dirs:
        if os.path.isfile(path+item):
            print(path+item)
            x = np.loadtxt(path+item, delimiter=',', unpack=True, dtype='float32')
            f, e = os.path.splitext(path+item)

            z_data = np.transpose(x[2])
            # specgram_z = stft.spectrogram(z_data)
            specgram_z = stft.spectrogram(z_data, window=0.4)
            plt._imsave(f + '.jpg', abs(specgram_z), vmin=-40, vmax=40, cmap=plt.get_cmap('coolwarm'), format='jpg') # gray Wistia


convert()
# '''


# 파일 하나만 스펙트럼 이미지로 바꿔줌
'''
import matplotlib.pyplot as plt
import stft
import numpy as np

x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',', unpack=True)
# toimage(x).show()
z_data = np.transpose(x[2])
specgram_z = stft.spectrogram(z_data, window=0.4)
plt._imsave('all(test).jpg', abs(specgram_z), vmin=-40, vmax=40, cmap=plt.get_cmap('coolwarm'), format='jpg')

# '''

