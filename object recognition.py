import numpy as np
import imutils
import cv2
prototxt="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
confThresh=0.2
CLASSES=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor","mobile"]
COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))
print("Loading model...")
net=cv2.dnn.readNetFromCaffe(prototxt,model)
print("Model training")
print("starting camera feed....")
cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frame=imutils.resize(frame,width=1000)
    (h,w)=frame.shape[:2]
    imResizeBlob=cv2.resize(frame,(300,300))
    blob=cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()
    print(detections)
    detshape=detections.shape[2]
    for i in np.arange(0,detshape):
        confidence=detections[0,0,i,2]
        if confidence>confThresh:
            idx=int(detections[0,0,i,1])
            print("CLASSID:",detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            if startY-15>15:
                y=startY-15
            else:
                startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
