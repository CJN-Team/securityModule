import cv2
import numpy as np
import face_recognition
import os
import pickle
# from PIL import ImageGrab
 
path = 'dataset'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # with open('dataset_names.dat', 'wb') as f:
    #     pickle.dump(classNames, f)

def findEncodings(images):
    print("start")
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(encodeList, f)



findEncodings(images)