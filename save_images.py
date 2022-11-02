import numpy as np
import cv2
import os
from imutils.video import VideoStream


# ime osobe
namePerson = "Tomislav_Horvat"
pathToSave = os.path.join("myDataset/", namePerson)
os.makedirs(pathToSave, exist_ok = True)
toAdd = 20


# video
cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)   

k = 0

while True:
    ret, frame = cap.read()
    
    if ret==False:
        break    
   
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite(os.path.join(pathToSave, str(k)+'.png'), frame)
        k = k + 1
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
