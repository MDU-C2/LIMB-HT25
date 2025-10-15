import depthai as dai
import cv2
import numpy as np

device = dai.Device()

print(f"Device: {device.getDeviceInfo()}")


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    
    cam = pipeline.create(dai.node.Camera).build()
    video_queue = cam.requestOutput((640, 400)).createOutputQueue()


    print("Pipeline created.")
    pipeline.start()

    while pipeline.isRunning():
        video_in = video_queue.get()
        assert isinstance(video_in, dai.ImgFrame)
        cv2.imshow("Camera", video_in.getCvFrame())
        
        if cv2.waitKey(1) == ord('q'):
            print("Exiting...")
            break

    pipeline.stop()
    cv2.destroyAllWindows()