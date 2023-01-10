import cv2
import numpy as np;


Face_cascade =cv2.CascadeClassifier('C://Users//ANI-IN//AppData//Local//Programs//Python//Python310//Lib//cv2//data//haarcascade_frontalface_default.xml')
Web_cam = cv2.VideoCapture(0)                                               #To enable webcam using cv2 library
recognizer = cv2.face.LBPHFaceRecognizer_create()                           #used to recognize the face of a person
recognizer.read('C:/Users/ANI-IN/Desktop/Project/trainer/trainer.yml')      #Getting the files that is already trained in part 2


id = 0
font = cv2.FONT_HERSHEY_SIMPLEX     #Font we are using 

while True:

    ret, Img_frame = Web_cam.read()                         #Reading the frame using read function if something goes wrong it will return false
    gray = cv2.cvtColor(Img_frame, cv2.COLOR_BGR2GRAY)      #Converting particular Image frame to grayscale & storing into gray variable
    faces = Face_cascade.detectMultiScale(gray, 1.3, 7);    #It will return x,y,w,h coordinates

    for (x, y, w, h) in faces:                              #Loops for each faces or particular frame
        roi_gray = gray[y:y + h, x:x + w]                   #Region of Interest 

        cv2.rectangle(Img_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)        #Creating rectangle around the face
        id, confidence = recognizer.predict(roi_gray)                           #matching the face with the region of Interest
        
        if (confidence < 75):
            if (id == 1):
                id = 'Shashank kothari'
                cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            elif (id == 2):
                id = 'Prof. (Dr.) Kamal Ghanshala'
                cv2.putText(Img_frame, id, (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            elif (id == 3):
                id = 'Rs rawat '
                cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            id = 'Unknown'
            cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
               
    cv2.imshow('Face Detection', Img_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   #press q to quit the window
        break

Web_cam.release()                           #stop using webcam 
cv2.destroyAllWindows()