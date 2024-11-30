
# Object Recognition using MobileNetSSD

This project demonstrates object recognition using the pre-trained MobileNetSSD model. The model identifies and classifies objects from a live camera feed.

## Requirements

Before running the project, make sure you have the following libraries and dependencies installed:

### Python Libraries:
- `opencv-python` (for capturing video feed and image processing)
- `tensorflow`
- `numpy`
- `imutils`

You can install the necessary dependencies using the following command:

```bash
pip install opencv-python tensorflow numpy imutils
```

## Model

The pre-trained MobileNetSSD model is used for object detection. You can download the MobileNetSSD model files from the following links:
- **Model file**: `mobilenetssd_deploy.caffemodel` (the pre-trained model)
- **Configuration file**: `mobilenetssd_deploy.protxt` (the model configuration)

## How to Run the Project

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Run the Object Detection Script:
To perform object recognition using the live camera feed, use the `object_recognition.py` script. The script will continuously detect and display objects in the camera feed.

```bash
python object_recognition.py --camera
```

### 3. Result:
The output will show the detected objects with bounding boxes around them and the class labels.

## Files Included:
- `mobilenetssd_deploy.caffemodel`: The pre-trained MobileNetSSD model.
- `mobilenetssd_deploy.protxt`: The model configuration file.
- `object_recognition.py`: The main script to perform object detection.
- `requirements.txt`: The list of dependencies for the project.


