Real-Time Object Detection System

This project demonstrates a real-time object detection system using TensorFlow's SSD MobileNet V2 model and OpenCV. The system captures video frames from a webcam, detects objects, and displays the results with bounding boxes and labels.

Features

- Real-time object detection using a pre-trained SSD MobileNet V2 model
- Bounding box and label display for detected objects
- Configurable logging of detection events
- Time-stamping of video feeds

Requirements

- Python 3.x
- OpenCV
- TensorFlow

Installation

1. Clone the repository:

2. Navigate to the project directory:cd x(name of your repo)

3. Install the required packages:

   pip install -r requirements.txt

Usage

1. Ensure your webcam is connected.

2. Run the detection script:

   python detect.py

3. Press 'q' to quit the application.

Project Structure

- detect.py: Main script for running the object detection system.
- models/ssd_mobilenet_v2/: Directory containing the pre-trained SSD MobileNet V2 model.
- detection_log.txt: Log file for recording detection events.

Logging

Detection events are logged in the detection_log.txt file with the following format:

YYYY-MM-DD HH:MM:SS - Label detected at coordinates: (xmin, ymin, xmax, ymax)

Example

When the script is running, the webcam feed will be displayed with bounding boxes and labels for detected objects. The current time will be shown in the top-left corner of the video feed.

License

This project is licensed under the MIT License.

Acknowledgements

- TensorFlow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection
- OpenCV: https://opencv.org/
