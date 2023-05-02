import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)

#maxhands - max number of hands that can be detected
detector = HandDetector(maxHands=1)                 

classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt")

#to save data for A/B/C
folder = "Data/C"
#counter = 0

labels = ["A" , "B" , "C"]


while (True):
    ret , img = cap.read()

    #making copy of img
    imgOutput = img.copy()

    #if hand is detected in the image
    #'hands' contain list of dictionaries i.e. each dictionary is assigned to each hand detected
    #Dic contains info like landmarks(to draw skeleton of hand) , bbox coordinates , type of hand(left/right)
    hands , img = detector.findHands(img)

    if hands:
        hand = hands[0]
        imgSize = 300

        #details of bounding box around hand 
        #(x,y) - coordinates of top left corner of bbox / w,h - width , height of bbox
        x,y,w,h = hand['bbox']

        #creating white background
        imgWhite = np.ones([imgSize,imgSize,3] , np.uint8)*255

        #croping the bounding box
        imgCrop = img[y-20:y+h+20 , x-20:x+w+20]

        #gives dimensions of imgcrop(i.e. dimensions of bbox)
        imgCropShape = imgCrop.shape


        #putting croped bounding box on white background - imgWhite is modified
        #imgWhite[0:imgCropShape[0] , 0:imgCropShape[1]] = imgCrop

        #to fit the croped bbox on 300x300 white backgrnd
        #if h>w stretch h to imgSize and modify w
        #if w>h stretch w to imgSize and modify h
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h                 #imgSize is imgWhite size
            wNew = math.ceil(k*w)         #to get integer value of width

            #resizing croped bbox to new width and height
            imgResized = cv2.resize(imgCrop , (wNew,imgSize))
            imgResizeShape = imgResized.shape

            wGap = math.ceil((imgSize-wNew)/2)

            #imgWhite[0:imgCropShape[0] , 0:imgCropShape[1]] = imgCrop
            #putting resized croped image on white backgrnd
            #imgWhite[0:imgResizeShape[0] , 0:imgResizeShape[1]] = imgResized
            imgWhite[0:imgResizeShape[0] , wGap:wGap+wNew] = imgResized

            #to get the predictions
            prediction , index = classifier.getPrediction(imgWhite,draw = False)
            print(prediction,index)

        else:
            k = imgSize/w
            hNew = math.ceil(k*h)
            imgResized = cv2.resize(imgCrop , (imgSize,hNew))
            imgResizeShape = imgResized.shape
            hGap = math.ceil((imgSize-hNew)/2)
            imgWhite[hGap:hGap+hNew , 0:imgResizeShape[1]] = imgResized
            prediction , index = classifier.getPrediction(imgWhite,draw = False) #draw false not to draw on imgWhite
            print(prediction,index)

        #putting lebels on the image
        cv2.putText(imgOutput , labels[index] , (x-18,y-28) , cv2.FONT_HERSHEY_COMPLEX , 2 , (0,255,0) , 2)

        cv2.rectangle(imgOutput , (x-25,y-20) , (x+w+25,y+h+20) , (0,255,0) , 5)

     #   cv2.imshow('imgCrop' , imgResized)
     #  cv2.imshow('imgWhite' , imgWhite)

        if cv2.waitKey == ord('q'):
            break
    if ret == True:
        cv2.imshow('image',imgOutput)
        cv2.waitKey(1)
    else:
        break


cap.release()
cv2.destroyAllWindows()

