from mtcnn import MTCNN
import cv2

def faceReader(imageAddress):
    img = cv2.imread(imageAddress)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    out = detector.detect_faces(img)

    for i in range(len(out)):
        x , y , w , h = out[i]['box']
        kp = out[i]['keypoints']
        for _ , value in kp.items():
            cv2.circle(img , value , 3 , (0,0,255) , -1)
        cv2.rectangle(img , (x , y) , (x+w , y+h) , (0,255,0) , 2)
        
    return img

img = faceReader('testImages/team.jpg')

cv2.imshow('image' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()