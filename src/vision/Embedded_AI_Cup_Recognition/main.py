# To do : pip install -r requirements.txt
# python main.py

#Configuration required to make the execute the code :
#Python v3.10.11
#DepthAI v2.28.0

#The best.pt and finally blob model were trained on a dataset of 250 images of a specific cup, a starbucks cup, 
#be carefull with that, If you want to test the model on an other type of cup, it would be less efficient.

import cv2
import depthai as dai
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BLOB_PATH = "cup.blob"
LABEL_NAME = "Cup"

if not Path(BLOB_PATH).exists():
    raise FileNotFoundError(f"❌ Missing file : {BLOB_PATH} is not there!")

print(f">>> Loading model {LABEL_NAME}...")
pipeline = dai.Pipeline()

# 1. Color Camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 2. Depth (Stereo) - OAK-D LITE
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# 480p is needed for depth on OAK-D Lite
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# 3. AI (YOLO)
# We are using the specific node YOLO (Please use the v2.28, more stable)
nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setConfidenceThreshold(0.5)
nn.input.setBlocking(False)
nn.setBoundingBoxScaleFactor(0.5)
nn.setDepthLowerThreshold(100)
nn.setDepthUpperThreshold(5000)

# Parameters YOLOv8 Nano
nn.setNumClasses(1)
nn.setCoordinateSize(4)
nn.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
nn.setAnchorMasks({"side26": [1,2,3], "side13": [3,4,5]})
nn.setIouThreshold(0.5)

# Links
camRgb.preview.link(nn.input)
stereo.depth.link(nn.inputDepth)

# Outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
nn.passthrough.link(xoutRgb.input)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
nn.out.link(xoutNN.input)

# --- EXECUTION ---
print("Connection to the OAK-D Lite...")
try:
    with dai.Device(pipeline) as device:
        # Configuration USB
        device.setIrLaserDotProjectorBrightness(0) #à commenter si plantage à cette ligne, modèle LITE n'ont pas de laser

        qRgb = device.getOutputQueue("rgb", 4, False)
        qDet = device.getOutputQueue("detections", 4, False)

        print(f"\n✅ Ready ! {LABEL_NAME}.")
        
        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()
            
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                
                if inDet is not None:
                    detections = inDet.detections
                    for d in detections:
                        # Coordonates pixel
                        x1 = int(d.xmin * 640)
                        y1 = int(d.ymin * 640)
                        x2 = int(d.xmax * 640)
                        y2 = int(d.ymax * 640)
                        
                        # Coordonates XYZ
                        x_mm = int(d.spatialCoordinates.x)
                        y_mm = int(d.spatialCoordinates.y)
                        z_mm = int(d.spatialCoordinates.z)

                        # Draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"X:{x_mm} Y:{y_mm} Z:{z_mm}", (x1, y1+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        
                        print(f"🎯 Cup finded : X={x_mm} Y={y_mm} Z={z_mm}")

                cv2.imshow("OAK-D Lite", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n❌ ERROR : {e}")