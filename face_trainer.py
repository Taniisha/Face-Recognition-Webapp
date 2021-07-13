import cv2
import os
import numpy as np
from PIL import Image    #used to open and read images
#PIL stands for Python Imaging Library
#It provides general image handling and lots of useful basic image operations
#like resizing, cropping, rotating, color conversion and much more.

recognition = cv2.face.LBPHFaceRecognizer_create()  #it creates a face recognition model
#it uses pixels intensities and the variation to map and recognise a particular phase
path = "dataset"

def getImagesWithId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []

    for image_path in imagePaths:
        faceImg = Image.open(image_path).convert('L')   #return of Image.open(image_path) will be PIL image object
                                                        #convert('L') will convert image into grayscale
        faceNp = np.array(faceImg,'uint8')   #converting image into numpy array
        print(image_path)     #dataset/User.1.2
        ID = os.path.split(image_path)[-1].split('.')[1]    #to extract id (i.e. 1) from image path 
        faces.append(faceNp)
        IDs.append(int(ID))
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return(np.array(IDs),faces)

Ids,faces = getImagesWithId(path)
recognition.train(faces,Ids)   #supervised learning
recognition.save('trainingData.yml')
cv2.destroyAllWindows()

