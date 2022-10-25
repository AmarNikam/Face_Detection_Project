import cv2
from random import randrange

face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinate = face_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 2)
        cv2.imshow('video capture', frame)
    k = cv2.waitKey(1)
    if k == 81 or k == 113:
        break

webcam.release()
print("complete")

