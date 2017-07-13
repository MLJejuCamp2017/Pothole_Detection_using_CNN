from scipy.misc import toimage  # 각 윈도우를 확인하는 뻘짓용
import imutils
import argparse
import time
import cv2

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# 콘솔창에서 따로 사용할 일이 있을때만 사용하는 코드
    # py -3.5 sliding_window.py --image images/1.jpg >> 이렇게 사용하면 된다.
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# 이미지를 로드하고 윈도우의 사이즈를 정한다
# image = cv2.imread(args["image"]) # 콘솔창에서 사용할 때 위의 코드와 같이 사용
image = cv2.imread('images/1.jpg')
(winW, winH) = (3, 128) # 윈도우의 크기를 정한다

# 윈도우가 이미지를 따라 스텝만큼 슬라이딩 한다
for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    # 피쳐를 추출하여 분류기로 전달하는 코드가 들어오는 곳. 이건 Inception v3가 하겠지

    # 아래는 윈도우 확인용
    # print(window.shape) # 26열 에서 정한 윈도우의 크기 그대로 나타난다.
    # toimage(window).show() # 각각의 윈도우의 모습을 보고싶다면 주석을 해제하면 되지만 걷잡을수 없는 일이 발.생.한.다. 컬러로 표현되니 조금 다를 수 있음
    # toimage(window).save('test.jpg') # 저장하여 윈도우를 확인할 수 있다.

    # 그냥 윈도우를 그리는 코드
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.025)

print(window)