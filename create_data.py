import cv2
import numpy as np
vid=cv2.VideoCapture(0)
X,Y=[],[]
while True:
    ret,frame=vid.read()
    cv2.rectangle(frame,(30,230),(230,430),(0,255,0),2)
    aoi=frame[230:430,30:230]
    aoi=cv2.cvtColor(aoi,cv2.COLOR_BGR2GRAY)
    aoi=cv2.resize(aoi,(64,64))
    aoi=aoi.reshape((64,64,1))
    aoi=aoi/255.0
    X.append(aoi)
    Y.append(0)
    cv2.imshow('Recording 0s',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
while True:
    ret,frame=vid.read()
    cv2.rectangle(frame,(30,230),(230,430),(0,255,0),2)
    aoi=frame[230:430,30:230]
    aoi=cv2.cvtColor(aoi,cv2.COLOR_BGR2GRAY)
    aoi=cv2.resize(aoi,(64,64))
    aoi=aoi.reshape((64,64,1))
    aoi=aoi/255.0
    X.append(aoi)
    Y.append(1)
    cv2.imshow('Recording 1s',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
print('No. of Samples Recorded:',len(Y))
X,Y=np.array(X),np.array(Y)
np.save('X.npy',X)
np.save('Y.npy',Y)