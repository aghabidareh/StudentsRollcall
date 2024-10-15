import cv2 
import numpy as np

import firebase_admin
from firebase_admin import credentials , storage
from firebase_admin import db

cred = credentials.Credentials('databaseServices.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'url',
    'storageBucket' : 'url'
})

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

file = open('encodeFile.p' , 'rb')
encodeListKnownWithId = pickle.load(file)
file.close()
encodeListKnown,studentId = encodeListKnownWithId

nodeType = 0
counter = 0
id = -1

while True:
    success , img = cap.read()
    
    imgS = cv2.resize(img , (0,0) None,0.25,0.25)
    imgS = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_reognition.face_locations(imgS)
    encodeCurFrane = face_recognition.face_encodings(imgS,faceCurFrame)
    
    for encodeFace,faceLoc in zip(encodeCurFrane,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            bbox = x1,y1,x2,y2
            img = cvzone.cornerRect(img,bbox,rt=0)
            id = studentId[matchIndex]            
            if counter == 0:
                counter = 1
                nodeType = 1
    if counter != 0:
        
        if counter == 1:
            
            studentInfo = db.refrence(f'Students/{id}').get()
            print(studentInfo)
        
        cv2.putText(img , str(studentInfo['attendence']) . (100,100) , cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(img , str(studentInfo['id']) . (150,150) , cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.putText(img , str(studentInfo['name']) . (250,250) , cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            
        cpinter += 0
    
    cv2.imshow('Face Attendence' , img)
    cv2.waitKey(1)
