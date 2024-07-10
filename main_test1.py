import cv2
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

# 配置日志记录
logging.basicConfig(filename='detection_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 加载预训练的SSD MobileNet V2模型
model_dir = 'models/ssd_mobilenet_v2/ssd_mobilenet_v2_coco_2018_03_29'
model = tf.saved_model.load(f'{model_dir}/saved_model')
infer = model.signatures['serving_default']

# 标签映射
category_index = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


# 函数：进行检测
def detect(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = infer(input_tensor)
    return detections


# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 检测的最小置信度
min_conf_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    height, width, _ = frame.shape
    detections = detect(frame)

    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()[0]

    for i in range(len(detection_boxes)):
        if detection_scores[i] > min_conf_threshold:
            box = detection_boxes[i] * np.array([height, width, height, width])
            (ymin, xmin, ymax, xmax) = box.astype("int")

            class_id = detection_classes[i]
            if class_id in category_index:
                label = category_index[class_id]
                if label == 'person':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                logging.info('%s detected at coordinates: (%d, %d, %d, %d)', label.capitalize(), xmin, ymin, xmax, ymax)

    # 获取当前时间并显示在左上角
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 显示结果
    cv2.imshow('python face detect program Bokai Wang', frame)

    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
