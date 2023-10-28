import cv2
import numpy as np
import face_recognition
from collections import defaultdict

# Opencv DNN
#from ultralytics import YOLO
#model = YOLO('yolov8x.pt')
#model.export(format ="onnx")
#model_path = 'yolov8x.onnx'
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(604, 604), scale=1/300)

# Load danh sách các loại đối tượng
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Danh sách đối tượng")
print(classes)

# Đường dẫn tới ảnh của bạn
image_path = ("pic/nguoivsdongvat/tranning/1y.jpg")

# Đọc ảnh từ file
frame = cv2.imread(image_path)
brightened_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)#Tăng cường độ sáng (Brightness Enhancement) nhận diện rõ vật thể hơn
normalized_frame = frame / 300.0  # Chuẩn hóa về [0, 1], Chuẩn hóa và thay đổi tỷ lệ pixel
blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)#Làm mờ hình ảnh có thể loại bỏ nhiễu

# Chuyển ảnh sang ảnh xám
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_frame = cv2.equalizeHist(gray_frame)#Cân bằng histogram:cải thiện độ tương phản của hình ảnh
gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
# Object Detection
#(class_ids, scores, bboxes) = model.detect(gray_frame, confThreshold=0.2, nmsThreshold=0.6)
# Áp dụng Non-Maximum Suppression
def non_max_suppression(boxes, scores, threshold):
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

# Sử dụng NMS trên kết quả object detection
class_ids, scores, bboxes = model.detect(gray_frame, confThreshold=0.2, nmsThreshold=0.6)

# Chọn ra các bounding box sau khi áp dụng NMS
keep = non_max_suppression(bboxes, scores, threshold=0.6)

# Tạo một từ điển để lưu trữ các đối tượng theo class
detected_objects_by_class = defaultdict(list)

# Hiển thị kết quả
# Hiển thị kết quả
'''
for idx in keep:
    (x, y, w, h) = bboxes[idx]
    class_name = classes[class_ids[idx]]
    confidence = scores[idx]

    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
'''

# Hiển thị kết quả
for idx in keep:
    (x, y, w, h) = bboxes[idx]
    class_name = classes[class_ids[idx]]
    confidence = scores[idx]

    detected_objects_by_class[class_name].append((x, y, w, h))

    # Print the information
    print(f"Detected object: {class_name}, Confidence: {confidence}")

    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ...

# Chọn ra các bounding box sau khi áp dụng NMS
keep = non_max_suppression(bboxes, scores, threshold=0.6)

# Hiển thị kết quả
detected_objects = len(keep)
print(f"Total detected objects: {detected_objects}")

# Hiển thị kết quả
for idx in keep:
    (x, y, w, h) = bboxes[idx]
    class_name = classes[class_ids[idx]]

    cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# In ra từng nhóm Detected object riêng
for class_name, objects in detected_objects_by_class.items():
    print(f"Detected objects of class '{class_name}':")
    for obj in objects:
        print(f"  Bounding Box: {obj}")

# Hiển thị kết quả
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


