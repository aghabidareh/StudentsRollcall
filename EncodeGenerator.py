import cv2
import face_recognition
import pickle
import os


folderPath = 'Images'
modePathList = os.listdir(folderPath)
imgList = []
studentId = []
for path in modePathList:
    imgList.append(cv2.imread(os.path.join(folderPath , path)))
    studentId.append(os.path.splittext(path)[0])
    
def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
    
encodeListKnown = findEncoding(imgList)
encodeListKnownWithId = [encodeListKnown,studentId]

file = open('encodeFile.p','wb')
pickle.dump(encodeListKnownWithId , file)
file.close()
    
