# Tello Drone Control with Face Detection

This Python script allows you to control a Tello drone using Python and perform real-time face detection using the OpenVINO toolkit. The script also includes a graphical user interface (GUI) for controlling the drone.

## Prerequisites

Before running the script, make sure you have the following installed:

- [DjItelloPy](https://github.com/damiafuentes/DJITelloPy): Python library for controlling the Tello drone.
- [OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html): Intel's toolkit for high-performance deep learning inference.
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library.
- [Pillow](https://pillow.readthedocs.io/en/stable/): Python Imaging Library that is used for displaying images in the GUI.
- [Tkinter](https://docs.python.org/3/library/tkinter.html): Python's standard GUI library.

## Usage

1. Connect your Tello drone to your computer.
2. Download the OpenVINO model files for face detection (`face-detection-adas-0001.xml` and `face-detection-adas-0001.bin`) or use your own trained model.
3. Update the `model_xml` and `model_bin` variables in the script with the paths to your model files.
4. Run the script using Python.

The GUI will appear, allowing you to control the drone with the following options:
- **Take Off**: Initiates the drone takeoff.
- **Land**: Lands the drone safely.
- **Adjust Drone Position**: Automatically adjusts the drone's position based on face detection.
- **Detect Face**: Performs real-time face detection using the OpenVINO model.

Make sure to close the application using the "Exit Application" button to safely stop the video stream and disconnect from the drone.

## Additional Notes

- The script uses the `djitellopy` library for Tello drone control and the OpenVINO toolkit for deep learning inference.
- Face detection results are displayed in a separate window.
- The code is written in Python and uses Tkinter for the GUI.

Feel free to customize and enhance the script according to your needs.

Enjoy controlling your Tello drone and experimenting with face detection!
