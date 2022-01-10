import cv2
import numpy as np

LP_DETECTION_CFG = {
    "weight_path": "./pre_model/yolov3_custom_final.weights",
    "classes_path": "./pre_model/classes.names",
    "config_path":  "./pre_model/yolov3_custom.cfg"
}

def detectPlate(image):
    weight_path = LP_DETECTION_CFG["weight_path"]
    config_path = LP_DETECTION_CFG["config_path"]

    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    # Load Model
    net = cv2.dnn.readNet(weight_path, config_path)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    pts_x = 0
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        pts = np.array(
            [(round(x), round(y)), (round(x + w), round(y)), (round(x + w), round(y + h)), (round(x), round(y + h))])
        pts_x = pts
        break

    return pts_x

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

