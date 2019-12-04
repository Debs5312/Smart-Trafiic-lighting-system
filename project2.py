from project1 import BatchNormalization,DarknetConv,DarknetResidual,DarknetBlock,Darknet,YoloConv,YoloOutput,yolo_boxes,yolo_nms,YoloV3,load_darknet_weights,transform_images

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import time

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

class_names = [c.strip() for c in open('coco.names').readlines()]
n_classes=len(class_names)

yolo = YoloV3(classes=n_classes)

load_darknet_weights(yolo,'yolov3.weights')

def decode_image(img_path):
    #img = cv2.imread(img_path)
    img = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img_test = transform_images(img, 416)
    return (img_test)

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if (class_names[int(classes.numpy()[i])]) == "person":
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
                              x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
        elif (class_names[int(classes.numpy()[i])]) == "car" or (class_names[int(classes.numpy()[i])]) == "truck" or (class_names[int(classes.numpy()[i])]) == "bus":
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
                              x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def read_cam():
    cap = cv2.VideoCapture('Cars moving on road Stock Footage - Free Download.mp4')
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count = frame_count+1
        if ret == True:
            cv2.imwrite("test.jpg",frame)
            img_path = "test.jpg"
            boxes, scores, classes, nums = yolo(decode_image(img_path))
            img= draw_outputs(frame, (boxes, scores, classes, nums), class_names)
            cv2.imshow("output",img)        
        if cv2.waitKey(1) & frame_count>20:
            break
    
    cap.release()
    cv2.destroyAllWindows()

read_cam()
