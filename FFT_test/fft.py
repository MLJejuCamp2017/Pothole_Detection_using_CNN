import numpy as np
import matplotlib.pyplot as plt
import stft
from matplotlib.pyplot import specgram

# xyz = np.loadtxt('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv', unpack=True, delimiter=',', dtype='float32')
# xyz = np.loadtxt('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/실험체.csv', unpack=True, delimiter=',', dtype='float32')
# xyz = np.loadtxt('/Users/User/OneDrive/센서로그/이건쓰레기야/log - 복사본.csv', unpack=True, delimiter=',', dtype='float32')
# xyz = np.loadtxt('/Users/User/OneDrive/센서로그/자전거/포트홀/csv/다듬다듬/pothole (12).csv', unpack=True, delimiter=',', dtype='float32')
xyz = np.loadtxt('/Users/User/OneDrive/센서로그/자동차/방지턱/speedbump (29).csv', unpack=True, delimiter=',', dtype='float32')
# xyz = np.loadtxt('/Users/User/OneDrive/센서로그/자전거/포트홀/csv/다듬다듬/pothole (50).csv', unpack=True, delimiter=',', dtype='float32')
x_data = np.transpose(xyz[0])  # 한 행에 X 가 들어감, 즉 all.csv 상태 그대로 한 열을 읽어옴
y_data = np.transpose(xyz[1])
z_data = np.transpose(xyz[2])
# '''

print('xyz.shape :', xyz.shape)
print('x_data shape :', x_data.shape)
print('y_data shape :', y_data.shape)
print('z_data shape: ', z_data.shape)



plt.plot(x_data)
plt.plot(y_data)
plt.plot(z_data)
plt.show()


specgram_x = stft.spectrogram(x_data)
specgram_y = stft.spectrogram(y_data)
specgram_z = stft.spectrogram(z_data)


# plt.imshow(abs(specgram_x), vmin=-100, vmax=100, cmap=plt.get_cmap('Wistia'))
# plt.show()
# plt.imshow(abs(specgram_y), vmin=-100, vmax=100, cmap=plt.get_cmap('Wistia'))
# plt.show()
# plt.imshow(abs(specgram_z), vmin=-80, vmax=80, cmap=plt.get_cmap('Wistia'))
# plt.show()
# plt._imsave('test5.jpg', abs(specgram_z), vmin=-40, vmax=40, cmap=plt.get_cmap('Wistia'), format='jpg')

# '''
#####################################################
sp = np.fft.fftn(xyz)
specgram(abs(sp), NFFT=128, Fs=1, noverlap=100, cmap='coolwarm')

print('sp shape : ', sp.shape)
print('specgram : ', specgram)

plt.show()


######################################################



'''
import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt

filename = '/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv'

data = np.loadtxt(filename, delimiter=',')
sensor_data = fft.fft(data)

print(sensor_data)
print(sensor_data.shape)

plt.plot(data) # 원본 그래프
plt.show()

plt.plot(sensor_data) # ? 어쨌든 fft 통과한 그래프
plt.show()

freq = fft.fftfreq(len(sensor_data)) # 뭔지 모르겠는 그래프
plt.plot(freq.real, abs(sensor_data))
plt.show()

'''


'''
import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt

filename = '/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv'

data = np.loadtxt(filename, delimiter=',', unpack=True)

x_data = data[0]
y_data = data[1]
z_data = data[2]

plt.plot(abs(data[0])) # 원본 그래프
# plt.plot(data[1])
# plt.plot(data[2])
plt.show()

x_sensor_data = fft.fft(abs(x_data))
y_sensor_data = fft.fft(y_data)
z_sensor_data = fft.fft(z_data)


plt.plot(x_sensor_data) # ? 어쨌든 fft 통과한 그래프
plt.show()
# plt.plot(y_sensor_data) # ? 어쨌든 fft 통과한 그래프
# plt.show()
# plt.plot(z_sensor_data) # ? 어쨌든 fft 통과한 그래프
# plt.show()

# freq = fft.fftfreq(len(sensor_data)) # 뭔지 모르겠는 그래프
# plt.plot(freq.real, abs(sensor_data))
# plt.show()


'''







