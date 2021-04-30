import cv2
from random import randrange

# Load pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an img to detect face
img = cv2.imread('aguero.jpg')

# Working with videos
# webcam = cv2.VideoCapture()

#convert image
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around face
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,randrange(256),0),2)

# print face coordinates
# print(face_coordinates)

cv2.imshow('Face detector',img)
cv2.waitKey()

print("Code completed")