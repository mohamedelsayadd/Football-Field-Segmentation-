# Football Field Segmentation Project

This project implements a YOLOv8-based model to segment football fields in images. It provides a fast and accurate solution for applications such as sports analytics. The repository contains a Jupyter Notebook that guides you through the setup, training, and evaluation process.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Project Setup](#project-setup)
3. [Training Results](#training-results)
4. [Model Prediction](#model-prediction)
5. [Additional Notes](#additional-notes)
6. [Original File](#original-file)

## Environment Setup

1. **Check GPU Availability:**

    ```python
    !nvidia-smi
    ```

2. **Install Required Libraries:**

    ```python
    !pip install roboflow
    !pip install ultralytics
    ```

## Project Setup

1. **Import Libraries and Initialize Roboflow:**

    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("tennis-ai").project("football-boxes")
    version = project.version(1)
    dataset = version.download("yolov8")
    ```

2. **Load and Train YOLOv8 Model:**

    ```python
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    model.train(data="/content/football-boxes-1/data.yaml", epochs=25, imgsz=640)
    ```

## Training Results

1. **View Confusion Matrix:**

    ```python
    from IPython.display import display, Image
    img_path = "/content/runs/segment/train3/confusion_matrix.png"
    display(Image(img_path, width=800))
    ```

2. **View Training Results:**

    ```python
    img_path2 = "/content/runs/segment/train3/results.png"
    display(Image(img_path2, width=900, height=500))
    ```

## Model Prediction

1. **Load the Best Model and Make Predictions:**

    ```python
    model = YOLO("/content/runs/segment/train3/weights/best.pt")
    pred = model.predict(source="/content/football-boxes-1/test/images", conf=0.25, save=True)
    ```

2. **Display Predicted Images:**

    ```python
    import glob
    from IPython.display import display, Image

    for image_path in glob.glob("/content/runs/segment/predict/*.jpg")[:20]:
        display(Image(filename=image_path))
        print("\n")
    ```

## Additional Notes

- Replace `"YOUR_API_KEY"` with your actual Roboflow API key.
- Ensure that dataset paths and image paths are correctly set according to your file structure.

## Thank You

