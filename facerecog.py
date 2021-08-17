import cv2
from random import randrange
trained_data = cv2.CascadeClassifier('face_data0.xml')

webcam = cv2.VideoCapture(0)

while  True:
    B = randrange(256)
    G = randrange(256)
    R = randrange(256)

    sucess, frame = webcam.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinate = trained_data.detectMultiScale(gray_img)


    for (x,y,w,h) in face_coordinate:
        cv2.rectangle(frame, (x , y), ( x, y), (B,G,R), 5)

    cv2.imshow("Hello" , frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
#Traning_video(https://youtu.be/XrCAvs9AePM)
#Traing_website(https://docs.opencv.org/4.5.2/dc/d88/tutorial_traincascade.html)