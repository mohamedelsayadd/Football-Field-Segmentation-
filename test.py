from ultralytics import YOLO
import os 

os.chdir(r"C:\Users\moham\Desktop\Computer Vision\YOLO\YOLOv8")

model = YOLO("best_football.pt")

model.predict(source="salah.mp4", conf=0.25, save=True, show=True)
