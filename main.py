from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keyboard import press
model=load_model('gestureDino.h5')
vid=cv2.VideoCapture(0)
while True:
    ret,frame=vid.read()
    cv2.rectangle(frame,(30,230),(230,430),(0,255,0),2)
    aoi=frame[230:430,30:230]
    aoi=cv2.cvtColor(aoi,cv2.COLOR_BGR2GRAY)
    aoi=cv2.resize(aoi,(64,64))
    aoi=aoi.reshape((64,64,1))
    aoi=aoi/255.0
    aoi=aoi.reshape((1,64,64,1))
    pred=model.predict(aoi)
    cv2.imshow('Play',frame)
    if pred[0]>0.95:
        press('space')
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break