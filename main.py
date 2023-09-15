from openvino.inference_engine import IENetwork, IECore
import cv2



# read webcam frame and process the face detection using openvino
def main():

    # read the model
    model_xml = 'face-detection-adas-0001.xml'
    model_bin = 'face-detection-adas-0001.bin'
    
    # Create an instance of the IECore class
    ie = IECore()

    # Read the network using the read_network method
    net = ie.read_network(model=model_xml, weights=model_bin)

    # Load the network onto the device
    exec_net = ie.load_network(network=net, device_name='CPU')

    # read the webcam
    cap = cv2.VideoCapture(0)

    # read the frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # resize the frame
        frame = cv2.resize(frame, (672, 384))

        # preprocess the frame
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
                xmin = int(obj[3] * frame.shape[3])
                ymin = int(obj[4] * frame.shape[2])
                xmax = int(obj[5] * frame.shape[3])
                ymax = int(obj[6] * frame.shape[2])
                cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # show the frame
        cv2.imshow('frame', original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    
