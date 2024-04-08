import cv2
import numpy as np
import face_recognition

imgTrain=face_recognition.load_image_file("image/Train.jpeg")
imgTrain=cv2.cvtColor(imgTrain,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("image/2.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

facelocTrain=face_recognition.face_locations(imgTrain)[0]
encodeTrain=face_recognition.face_encodings(imgTrain)[0]
cv2.rectangle(imgTrain,(facelocTrain[3],facelocTrain[0]),(facelocTrain[1],facelocTrain[2]),(255,0,255),2)

facelocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

result=face_recognition.compare_faces([encodeTrain],encodeTest)
faceDis=face_recognition.face_distance([encodeTrain],encodeTest)
print(result,faceDis)

cv2.putText(imgTest,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Train Image",imgTrain)
cv2.imshow("Test Image",imgTest)
cv2.waitKey(0)