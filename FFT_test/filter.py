import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import toimage


def fourier():

    x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all.csv", delimiter=',')
    img = toimage(x)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows = img.height
    cols = img.width

    crow, ccol = int(rows/2), int(cols/2)

    # https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220568857153&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0  # 주파수 영역의 이미지 정중앙의 60x60 크기 영역에 있는 값을 모두 0으로 만듦
    f_ishift = np.fft.ifftshift(fshift)  # 역 쉬프트 함수를 이용해 재배열된 주파수들의 값을 원위치 시킴
    img_back = np.fft.ifft2(f_ishift)  # 역 푸리에변환을 하여 원래 이미지영역으로 전환
    img_back = np.abs(img_back)

    # plt.subplot(131), plt.imshow(img, cmap='gray')
    # plt.title('original image'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(132)
    # plt.imshow(img_back, cmap='gray')
    # plt.title('After HPF')
    # plt.xticks([])
    # plt.yticks([])

    plt._imsave('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/filtered_all_test.jpg', img_back, cmap='gray', format='jpg')

fourier()