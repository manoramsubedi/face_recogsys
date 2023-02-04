import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) #splitting extension of image
print(classNames)

#function to compute all encodings
def findEncodings(iamges):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #finding encoding
        encode = face_recognition.face_encodings(img)[0]
        #appending encode to the list
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))