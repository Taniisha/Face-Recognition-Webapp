import time
import numpy as np
import cv2
import os
#import tensorflow
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#Keras ImageDataGenerator - It lets you augment your images in real-time while your model is still training!
#                           You can apply any random transformations on each training image as it is passed to the model.

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
Id = input("Enter your Id - ")
sampleNum = 0

while(True):
    ret,frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(gray)
            if(len(eyes)>=2):
                cv2.imwrite("dataset/User." + Id + "." + str(sampleNum) + ".jpg" ,roi)
                sampleNum += 1
                cv2.imshow("image",roi)
        if(cv2.waitKey(100) == ord('q')):
            break
        elif(sampleNum > 30):
            break

cap.release()
cv2.destroyAllWindows()



