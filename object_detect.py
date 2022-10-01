import cv2
import numpy as np
from NMS import NMS

net = cv2.dnn.readNetFromONNX("yolov5n.onnx")
file = open("coco.txt","r")
classes = file.read().split('\n')

img = cv2.imread('test.jpg')
img = cv2.resize(img, (1100,650))
blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
net.setInput(blob)
detections = net.forward()[0]


# cx,cy , w,h, confidence, 80 class_scores
# class_ids, confidences, boxes

classes_ids = []
confidences = []
boxes = []
rows = detections.shape[0]

img_width, img_height = img.shape[1], img.shape[0]
x_scale = img_width/640
y_scale = img_height/640

for i in range(rows):
    row = detections[i]
    confidence = row[4]
    if confidence > 0.5:
        classes_score = row[5:]
        ind = np.argmax(classes_score)
        if classes_score[ind] > 0.5:
            classes_ids.append(ind)
            confidences.append(confidence)
            cx, cy, w, h = row[:4]
            x1 = int((cx- w/2)*x_scale)
            y1 = int((cy-h/2)*y_scale)
            x2 = int(w * x_scale) + x1
            y2 = int(h * y_scale) + y1
            box = np.array([x1,y1,x2,y2])
            boxes.append(box)
boxes = np.array(boxes)
confidences = np.array(confidences)

boxes, confidences = NMS(boxes, confidences)

for i in range(len(boxes)):
    x1,y1, x2, y2 = boxes[i]
    label = classes[classes_ids[i]]
    conf = confidences[i]
    text = label + "{:.2f}".format(conf)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
    cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),1)

cv2.imshow("IMAGE",img)

k = cv2.waitKey(0)
