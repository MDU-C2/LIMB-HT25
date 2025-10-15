import depthai as dai
import cv2
import numpy as np

device = dai.Device()

print(f"Device: {device.getDeviceInfo()}")

cam_features = {}
for cam in device.getConnectedCameraFeatures():
    cam_features[cam.socket] = (cam.width, cam.height)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    output_queues = {}
    camera_sensors = device.getConnectedCameraFeatures()
    
    for sensor in camera_sensors:
        cam = pipeline.create(dai.node.Camera).build(sensor.socket)

        request_resolution = (
            (sensor.width, sensor.height)
            if sensor.width <= 1920 and sensor.height <= 1080
            else (1920, 1080)
        ) # Limit frame size to 1080p

        cam_out = cam.requestOutput(
            request_resolution, dai.ImgFrame.Type.NV12, fps=30
        ).createOutputQueue()
        output_queues[str(sensor.socket)] = cam_out
    
    print("Pipeline created.")
    pipeline.start()
   
    while pipeline.isRunning():
        for name in output_queues.keys():
            queue = output_queues[name]
            
            video_in = queue.get()
            assert isinstance(video_in, dai.ImgFrame)
           
            cv2.imshow(name, video_in.getCvFrame())
            if cv2.waitKey(1) == ord('q'):
                break

    pipeline.stop()
