import sys
import os
import numpy as np
import cv2
from sface import SFace
from imutils import paths
sys.path.append('opencv_zoo/models/face_detection_yunet')
from yunet import YuNet
import numpy as np
from datetime import datetime
import pickle
from jetcam.csi_camera import CSICamera

#funkcija za zapisivanje osoba i vremena prve detekcije:
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

myDataset = "dataset za probu/"

recognizer = SFace(modelPath='opencv_zoo/models/face_recognition_sface/face_recognition_sface_2021dec.onnx', disType=0)
detector = YuNet(modelPath='opencv_zoo/models/face_detection_yunet/face_detection_yunet_2021dec.onnx',
                inputSize=[320, 320],
                confThreshold=0.8,
                nmsThreshold=0.3,
                topK=5000)

imagePaths = list(paths.list_images(myDataset))
nameFeatureList = []

# za svaku sliku izracunaj feature
print("[INFO] Pronadjeno " + str(len(imagePaths)) + " slika.")
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    img = cv2.imread(imagePath)
    detector.setInputSize([img.shape[1], img.shape[0]])
    face = detector.infer(img)

    if face is not None:
        print("Pronadjeno lice za " + name)
        feature = recognizer.infer(img, face)
        nameFeatureList.append([name, feature])

camera = CSICamera(width=425,height=240,framerate=30)

camera.running = True



cv2.namedWindow("Frame", cv2.WINDOW_GUI_EXPANDED)
while True:
    frame = camera.value.copy()
    frame = cv2.flip(frame,0)
    matchedDict = {}
    detector.setInputSize([frame.shape[1], frame.shape[0]])
    faces = detector.infer(frame)

        # ako je barem jedno lice na okviru
    if faces is not None:
        noFaces = faces.shape[0]
            
            # idi kroz sva lica i usporedi s bazom
        for i in range(0, noFaces):
            featureNew = recognizer.infer(frame, faces[i])
                # idi kroz sva pohranjena lica
            for f_name in nameFeatureList:
                score = recognizer.compareTwoFeatures(featureNew, f_name[1])
                if score > recognizer._threshold_cosine:
                    if f_name[0] in matchedDict:
                        if matchedDict[f_name[0]][1] < score:
                            matchedDict[f_name[0]][1] = score
                    else:
                        matchedDict[f_name[0]] = [faces[i], score]


            # prikazi detekcije i imena
        for name in matchedDict:
            (startX, startY, width, height) = matchedDict[name][0][0:4].astype(int)
            cv2.rectangle(frame, (startX, startY), (startX+width, startY+height),  (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY - 35), (startX+width, startY), (0, 255, 0), cv2.FILLED)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, "{} / {:1.2f}".format(name, matchedDict[name][1]), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1)
                #pozovi funkciju za zapisivanje imena u .csv
            markAttendance(name)

        cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
      
            break
    

