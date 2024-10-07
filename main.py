from mtcnn import MTCNN
import cv2

cap = cv2.VideoCapture('testImages/maryam.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

detector = MTCNN()

while (cap.isOpened()):
    ret , img = cap.read()
    
    if ret == True:
    
        if img is None:break
    
        img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
    
        try:
    
            out = detector.detect_faces(img)
    
            x , y , w , h = out['box']
    
            cv2.rectangle(img , (x , y) , (x+w , y+h) , (0 , 255 , 0) , 2)
            cv2.imshow('face' , img)
            if cv2.waitKey(30) == ord('q'):
                break
            
        except:
            pass
       
cap.release()

cv2.destroyAllWindows()