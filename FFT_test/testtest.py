# import numpy as np
# import cv2
# from scipy.misc import toimage
# import matplotlib.pyplot as plt
#
# def hist2D():
#     img = cv2.imread("/tf_files/sensor_log(grayscale)/asphalt/asphalt (13).jpg")
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#     # hist = cv2.calcHist([hsv], [0, 1], None, [None, None], [0, None, 0, None])
#
#     plt.imshow(hist, interpolation='nearest')
#     plt.show()
#
#     cv2.imshow('hist2D', hist)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# hist2D()



import numpy as np
import cv2

hscale = 10

def onChange(x):
    global hscale

    hscale = x

def HSVmap():
    hsvmap = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsvmap.shape[:2])

    hsvmap[:,:,0] = h
    hsvmap[:,:,1] = s
    hsvmap[:,:,2] = 255

    hsvmap = cv2.cvtColor(hsvmap, cv2.COLOR_HSV2BGR)

    return hsvmap

def hist2D():
    img = cv2.imread('/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/all(spectrum).jpg')

    hsvmap = HSVmap()

    cv2.namedWindow('hist2D', 0)
    cv2.createTrackbar('scale', 'hist2D', hscale, 32, onChange)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])

        hist = np.clip(hist*0.005*hscale, 0, 1)
        hist = hsvmap*hist[:,:,np.newaxis] / 255.0

        cv2.imshow('hist2D', hist)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    cv2.destroyAllWindows()

hist2D()