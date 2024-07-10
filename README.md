# Real-Time Object Detection System

This project demonstrates a real-time object detection system using TensorFlow's SSD MobileNet V2 model and OpenCV. The system captures video frames from a webcam, detects objects, and displays the results with bounding boxes and labels.

## Features

- **Real-time Object Detection**: Utilizes a pre-trained SSD MobileNet V2 model for real-time object detection.
- **Bounding Box and Label Display**: Draws bounding boxes and labels around detected objects.
- **Configurable Logging**: Logs detection events with detailed information.
- **Time-Stamping**: Displays the current time on the video feed.

![sample](https://github.com/kairwang01/Computer-Vision-python/assets/38762041/a32322ab-3f6a-438b-a73c-a34a52d58244)

## Requirements

- Python 3.x
- OpenCV
- TensorFlow

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kairwang01/Computer-Vision-python.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd name of the directory
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Ensure your webcam is connected**.

2. **Run the detection script**:

   ```bash
   python detect.py
   ```

3. **Press 'q' to quit the application**.

## Project Structure

- `detect.py`: Main script for running the object detection system.
- `models/ssd_mobilenet_v2/`: Directory containing the pre-trained SSD MobileNet V2 model.
- `detection_log.txt`: Log file for recording detection events.

## Logging

Detection events are logged in the `detection_log.txt` file with the following format:

```
YYYY-MM-DD HH:MM:SS - Label detected at coordinates: (xmin, ymin, xmax, ymax)
```

## Example

When the script is running, the webcam feed will be displayed with bounding boxes and labels for detected objects. The current time will be shown in the top-left corner of the video feed.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [OpenCV](https://opencv.org/)
