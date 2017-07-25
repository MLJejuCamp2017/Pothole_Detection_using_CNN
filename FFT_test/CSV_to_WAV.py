# # CSV 파일을 WAV 파일로 변환
#
# import numpy as np
# from scipy.io import wavfile
# from scipy.signal import resample
#
# data = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',')
# data_resampled = resample(data, 4000)
#
# wavfile.write('output.wav', 4000, data_resampled)
#




import scipy.io.wavfile
import numpy as np
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display

data = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',')
scipy.io.wavfile.write("karplus.wav", 4000, data)

x, sr = librosa.load('/Users/User/PycharmProjects/network/ML_Camp/FFT_test/karplus.wav')
ipd.Audio(x, rate=sr)

hop_length = 512
n_fft = 2048
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

float(hop_length)/sr # units of seconds

float(n_fft)/sr  # units of seconds

S = abs(X)**2
librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='linear')

# X.shape