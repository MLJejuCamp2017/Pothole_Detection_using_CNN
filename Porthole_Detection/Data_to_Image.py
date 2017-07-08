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

import numpy as np
from scipy.misc import toimage

x = np.loadtxt("/Users/User/PycharmProjects/network/ML_Camp/Porthole_Detection/test/original/porthole_test.csv", delimiter=',')
# toimage(x).show()
toimage(x).save('porthole_test.jpg')
