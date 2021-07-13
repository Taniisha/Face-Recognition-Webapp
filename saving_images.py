import time
import numpy as np
import cv2
import os
import tensorflow
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
userName = input()
sampleNum = 0


while(True):
    ret,frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(gray)
            if(len(eyes)>=2):
                print("face detected")
                cv2.imshow("image",roi)
                t = time.strftime("%Y-%m-%d_%H-%M-%S")
                print("Image"+t+"Saved")
                path = "C:/Users/Tanishq Jain/Desktop/OpenCV_projects/Face_recognition/images"
                print("try")
                if(cv2.waitKey(1) & 0xff==ord('c')):
                    print("reaching")
                    #cv2.imwrite(os.path.join(path,userName+".jpg"),roi)
                    cv2.imwrite(os.path.join(path,t+".jpg"),roi)
                    print("saved this")
                    datagen = ImageDataGenerator(
                                 rotation_range = 40,
                                 width_shift_range = 0.2,
                                 height_shift_range = 0.2,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True,
                                 fill_mode = 'nearest')
                    #img = load_img("C:/Users/Tanishq Jain/Desktop/OpenCV_projects/Face_recognition/images/"+userName+".jpg")
                    img = load_img("C:/Users/Tanishq Jain/Desktop/OpenCV_projects/Face_recognition/images/"+t+".jpg")
                    #img = load_img(path + t +".jpg")
                    ita = img_to_array(img)
                    ita = ita.reshape((1,) + ita.shape)
                    os.makedirs("./users/"+userName,exist_ok=True)

                    i = 0
                    for batch in datagen.flow(ita,batch_size=1,save_to_dir="./users/"+userName,save_prefix="face",save_format="jpeg"):
                        i += 1
                        print('i=',i)
                        if(i>20):
                            print("Done augmented images")
                            break
                else:
                    print("not pressed")
        print("q")
        if(cv2.waitKey(1) & 0xff==ord("q")):
            print("q is pressed")
            break
    else:
        print("False")
        
cap.release()
cv2.destroyAllWindows()



