#from multiprocessing.resource_sharer import stop
import cv2
from random import randrange
#Load pre-trained data from opencv xml file
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose image to detect face 
#img = cv2.imread('rdj.png') #this is how you upload an image

#capture video from cam
cam = cv2.VideoCapture(0) #0 in bracket makes default but can replace it with a file video to scan

while True:

    frame_read, colour_img = cam.read()##reads fram
   
    #make img gray so colour is one number and not rgb which is easier for program
    gscale_img = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gscale_img)

    for (x,y,w,h) in face_coordinates:
     cv2.rectangle(colour_img,(x, y),(x+w, y+h), (0,256,0),7)


    cv2.imshow('Face Detector' , colour_img)
    stop_key = cv2.waitKey(1)      
        
    if stop_key == 27:
        break
#wkey = cv2.waitKey(1)

cam.release()

'''
#detect Faces
face_coordinates = trained_face_data.detectMultiScale(gscale_img)

#draw rectangle around face 
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x, y),(x+w, y+h), (randrange(256),randrange(256),randrange(256)),7)


cv2.imshow('Face Detector' , img)
 #used to wait before terminating program. without this, image is shown for a second and continues to end the program
 #keeps image open
cv2.waitKey()
'''
