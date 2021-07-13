import streamlit as st
import cv2
import numpy as np
from PIL import Image

face_cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainingData.yml')

def detectFaces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(gray, 1.3, 5)
    name = "Unknown"
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi = gray[y:y + h, x:x + w]
        id, uncertainty = rec.predict(roi)
        print(id, uncertainty)

        if(uncertainty < 53):
            if(id==1 ):
                name = "Tanisha"
                cv2.putText(img,name,(x,y+h),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(0,0,255))
        else:
            cv2.putText(img,"Unknown", (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
    return(img)

def main():
    """Face Recognition App"""

    st.title("Streamlit Tutorial")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img = detectFaces(Image.open(image_file))
        st.image(result_img)


if __name__ == '__main__':
    main()







