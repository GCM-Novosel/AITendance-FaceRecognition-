import keras
from tensorflow import keras
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import color
import matplotlib.image as mpimg
import numpy as np
import cv2

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_SIMPLEX

model = keras.models.load_model('MD/')

cp = cv2.VideoCapture(0)
kernel1 = np.ones((7,7), np.uint8)
kernel2 = np.ones((5,5), np.uint8)

pad = 15
size_th = 32
mnist_size = 28

while True:
    ret, frame = cp.read(0)

    img_vector = cropped_digit.reshape(1, 200, 220, 3)

    label = np.argmax(model.predict(img_vector))
    label = str(int(label))

    cv2.rectangle(frame, (x - pad, y - pad), (x + pad + w, y + pad + h), color=(255, 255, 0))

    cv2.putText(frame, label, (rect[0], rect[1]), font,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # show results
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break