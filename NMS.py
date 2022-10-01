'''
1- Take the bounding boxes, confidence_scores and set the iou threshold. Extract all the
   coordinates of top-left and bottom-right corners in the separate variables.

2- Calculate the areas of all the bounding boxes (detections).

3- Select the detection(bounding box) 'A' with highest confidence score and remove it from the array
   of detections 'Boxes' and add it to the final detection list 'keep'. ('keep' is initially empty).

4- Calculate the IoU of this detection 'A' with every other detections in 'Boxes'. If the IoU is greater
   than the iou_threshold for any detection present in 'Boxes', remove that detection from 'Boxes'.

5- If there are still detections left in 'Boxes', then go to Step 2 again, else return the list
   'keep' containing the filtered bounding boxes.
'''

import numpy as np

def NMS(boxes, conf_scores, iou_thresh = 0.5):

    # boxes [[x1,y1, x2,y2],[x1,y1, x2,y2].....]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences
