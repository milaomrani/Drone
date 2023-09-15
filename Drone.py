from djitellopy import Tello
from openvino.inference_engine import IECore
import cv2

def get_distance_to_face(obj, frame_width, frame_height):
    # calculate the width and height of the box
    box_width = obj[5] - obj[3]
    box_height = obj[6] - obj[4]
    
    # Using the width and height of the box to estimate the distance 
    # (this is a simple estimation, you might need to adjust the coefficient)
    distance = 5000 * (0.5 / (box_width * box_height))
    return distance

def adjust_drone_position(tello, distance):
    if distance > 50:
        tello.move_forward(min(int(distance - 50), 100))
    elif distance < 50:
        tello.move_back(min(int(50 - distance), 100))


def main():
    # Initialize the Tello drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    # read the model
    model_xml = 'face-detection-adas-0001.xml'
    model_bin = 'face-detection-adas-0001.bin'
    
    # Create an instance of the IECore class
    ie = IECore()

    # Read the network using the read_network method
    net = ie.read_network(model=model_xml, weights=model_bin)

    # Load the network onto the device
    exec_net = ie.load_network(network=net, device_name='CPU')

    while True:
        # Get the frame from Tello drone
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (672, 384))

        # preprocess the frame
        input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))
        n, c, h, w = net.input_info[input_blob].input_data.shape
        original_frame = frame.copy()

        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((n, c, h, w))

        # inference
        res = exec_net.infer(inputs={input_blob: frame})

        # postprocess
        res = res[out_blob]
        for obj in res[0][0]:
            if obj[2] > 0.5:
                # xmin = int(obj[3] * original_frame.shape[1])
                # ymin = int(obj[4] * original_frame.shape[0])
                # xmax = int(obj[5] * original_frame.shape[1])
                # ymax = int(obj[6] * original_frame.shape[0])
                # cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                xmin = int(obj[3] * original_frame.shape[1])
                ymin = int(obj[4] * original_frame.shape[0])
                xmax = int(obj[5] * original_frame.shape[1])
                ymax = int(obj[6] * original_frame.shape[0])
                cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Get distance to the face and adjust drone position
                distance = get_distance_to_face(obj, original_frame.shape[1], original_frame.shape[0])
                adjust_drone_position(tello, distance)

        # show the frame
        cv2.imshow('frame', original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop the video stream and disconnect
    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
