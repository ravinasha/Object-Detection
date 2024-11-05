import cv2
import numpy as np
import time

# Load the YOLO model
net = cv2.dnn.readNet("Real-Time Object Detection OpenCV Python Source Code/weights/yolov3-tiny.weights", "Real-Time Object Detection OpenCV Python Source Code/configuration/yolov3-tiny.cfg")
with open("Real-Time Object Detection OpenCV Python Source Code//configuration//coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layer_names = [layer_names[idx - 1] for idx in output_layers_indices]

#outs = net.forward(output_layer_names)


colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0

while True:
    # Read webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Visualizing data
    
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Press [Esc] to exit", (10, 80), font, 0.7, (0, 255, 255), 2)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        print("[button pressed] ///// [esc].")
        print("[feedback] ///// Video capturing successfully stopped")
        break

cap.release()
cv2.destroyAllWindows()
