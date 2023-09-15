from djitellopy import Tello
from openvino.inference_engine import IECore
import cv2
from tkinter import *
from PIL import Image, ImageTk

last_detected_obj = None

def get_distance_to_face(obj, frame_width, frame_height):
    box_width = obj[5] - obj[3]
    box_height = obj[6] - obj[4]
    distance = 5000 * (0.5 / (box_width * box_height))
    return distance

frame_width, frame_height = 672, 384

def perform_inference(frame):
    global last_detected_obj
    # preprocessing and inference steps
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    frame = frame.transpose((2, 0, 1))
    frame = frame.reshape((n, c, h, w))
    res = exec_net.infer(inputs={input_blob: frame})
    
    # postprocess
    res = res[out_blob]
    for obj in res[0][0]:
        if obj[2] > 0.5:
            last_detected_obj = obj
    return res

def adjust_drone_position():
    global tello, last_detected_obj
    if last_detected_obj is None:
        return
    distance = get_distance_to_face(last_detected_obj, frame_width, frame_height)
    if distance > 50:
        tello.move_forward(min(int(distance - 50), 100))
    elif distance < 50:
        tello.move_back(min(int(50 - distance), 100))

def update_frame():
    global tello
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (frame_width, frame_height))
    perform_inference(frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    window.after(10, update_frame)

def take_off():
    global tello
    tello.takeoff()

def land():
    global tello
    tello.land()

def exit_application():
    global tello
    tello.streamoff()
    window.destroy()

def detect_face():
    global tello
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (672, 384))
    res = perform_inference(frame)
    
    original_frame = frame.copy()
    for obj in res[0][0]:
        if obj[2] > 0.5:
            xmin = int(obj[3] * original_frame.shape[1])
            ymin = int(obj[4] * original_frame.shape[0])
            xmax = int(obj[5] * original_frame.shape[1])
            ymax = int(obj[6] * original_frame.shape[0])
            cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', original_frame)


# Initialize the drone and OpenVINO model
tello = Tello()
tello.connect()
tello.streamon()

model_xml = 'face-detection-adas-0001.xml'
model_bin = 'face-detection-adas-0001.bin'

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name='CPU')

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape

# Initialize the GUI
window = Tk()
window.title("Tello Drone Control")

panel = Label(window)
panel.pack(pady=10)

btn_takeoff = Button(window, text="Take Off", command=take_off)
btn_takeoff.pack(side=LEFT, padx=10)
btn_land = Button(window, text="Land", command=land)
btn_land.pack(side=LEFT, padx=10)
btn_adjust_position = Button(window, text="Adjust Drone Position", command=adjust_drone_position)
btn_adjust_position.pack(side=LEFT, padx=10)
btn_detect_face = Button(window, text="Detect Face", command=detect_face)
btn_detect_face.pack(side=LEFT, padx=10)


btn_exit = Button(window, text="Exit Application", command=exit_application)
btn_exit.pack(side=RIGHT, padx=10)


# Start the GUI
update_frame()
window.mainloop()

# Ensure to stop the video stream and disconnect when closing the GUI
tello.streamoff()
