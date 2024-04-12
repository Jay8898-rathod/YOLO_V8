from ultralytics import YOLO
import numpy

#Loading the yolo pretrained model 
model = YOLO("yolov8n.pt", "v8")

#Prediction on image
detection_output = model.predict(source="C:\\Users\\Jay Rathod\\Desktop\\New Projects\\YOLO_V8\\data\\messi.jpg", conf=0.5, save=False)

#Display tensor array
print(detection_output)

#display numpy array 
print(detection_output[0].numpy())