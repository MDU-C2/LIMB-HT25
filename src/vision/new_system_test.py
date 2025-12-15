import argparse
import depthai as dai
import numpy as np
from depthai_nodes.utils import AnnotationHelper
from depthai_nodes import PRIMARY_COLOR, SECONDARY_COLOR
from typing import List
from ultralytics import YOLO

class KalmanFilter:
    def __init__(self, acc_std, meas_std, z, time):
        self.dim_z = len(z)
        self.time = time
        self.acc_std = acc_std
        self.meas_std = meas_std

        # the observation matrix
        self.H = np.eye(self.dim_z, 3 * self.dim_z)

        self.x = np.vstack((z, np.zeros((2 * self.dim_z, 1))))
        self.P = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        i, j = np.indices((3 * self.dim_z, 3 * self.dim_z))
        self.P[(i - j) % self.dim_z == 0] = (
            1e5  # initial vector is a guess -> high estimate uncertainty
        )

    def predict(self, dt):
        # the state transition matrix -> assuming acceleration is constant
        F = np.eye(3 * self.dim_z)
        np.fill_diagonal(F[: 2 * self.dim_z, self.dim_z :], dt)
        np.fill_diagonal(F[: self.dim_z, 2 * self.dim_z :], dt**2 / 2)

        # the process noise matrix
        A = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        np.fill_diagonal(A[2 * self.dim_z :, 2 * self.dim_z :], 1)
        Q = self.acc_std**2 * F @ A @ F.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        if z is None:
            return

        # the measurement uncertainty
        R = self.meas_std**2 * np.eye(self.dim_z)

        # the Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(3 * self.dim_z)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T


class KalmanFilterNode(dai.node.HostNode):
    
    def __init__(self):
        self._kalman_filters = {}
        super().__init__()

    def build(
        self,
        rgb: dai.Node.Output,
        tracker_out: dai.Node.Output,
        baseline: float,
        focal_length: float,
        label_map: List[str],
    ) -> "KalmanFilterNode":
        self.link_args(rgb, tracker_out)
        self._baseline = baseline
        self._focal_length = focal_length
        self._label_map = label_map
        return self

    def process(self, img_frame: dai.Buffer, tracklets: dai.Buffer) -> None:
        assert isinstance(img_frame, dai.ImgFrame)
        assert isinstance(tracklets, dai.Tracklets)
        frame: np.ndarray = img_frame.getCvFrame()
        current_time = tracklets.getTimestamp()

        annotation_helper = AnnotationHelper()

        for t in tracklets.tracklets:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            x_space = t.spatialCoordinates.x
            y_space = t.spatialCoordinates.y
            z_space = t.spatialCoordinates.z

            meas_vec_bbox = np.array(
                [[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]]
            )
            meas_vec_space = np.array([[x_space], [y_space], [z_space]])
            meas_std_space = z_space**2 / (self._baseline * self._focal_length)

            if t.status.name == "NEW":
                # Adjust these parameters
                acc_std_space = 10
                acc_std_bbox = 0.1
                meas_std_bbox = 0.05

                self._kalman_filters[t.id] = {
                    "bbox": KalmanFilter(
                        meas_std_bbox, acc_std_bbox, meas_vec_bbox, current_time
                    ),
                    "space": KalmanFilter(
                        meas_std_space, acc_std_space, meas_vec_space, current_time
                    ),
                }

            else:
                dt = current_time - self._kalman_filters[t.id]["bbox"].time
                dt = dt.total_seconds()
                self._kalman_filters[t.id]["space"].meas_std = meas_std_space

                if t.status.name != "TRACKED":
                    meas_vec_bbox = None
                    meas_vec_space = None

                if z_space == 0:
                    meas_vec_space = None

                self._kalman_filters[t.id]["bbox"].predict(dt)
                self._kalman_filters[t.id]["bbox"].update(meas_vec_bbox)

                self._kalman_filters[t.id]["space"].predict(dt)
                self._kalman_filters[t.id]["space"].update(meas_vec_space)

                self._kalman_filters[t.id]["bbox"].time = current_time
                self._kalman_filters[t.id]["space"].time = current_time

                vec_bbox = self._kalman_filters[t.id]["bbox"].x
                vec_space = self._kalman_filters[t.id]["space"].x

                x1_filter = (vec_bbox[0] - vec_bbox[2] / 2) / img_frame.getWidth()
                x2_filter = (vec_bbox[0] + vec_bbox[2] / 2) / img_frame.getWidth()
                y1_filter = (vec_bbox[1] - vec_bbox[3] / 2) / img_frame.getHeight()
                y2_filter = (vec_bbox[1] + vec_bbox[3] / 2) / img_frame.getHeight()

                annotation_helper.draw_rectangle(
                    top_left=(x1_filter, y1_filter),
                    bottom_right=(x2_filter, y2_filter),
                    thickness=2,
                    outline_color=PRIMARY_COLOR,
                )
                annotation_helper.draw_text(
                    text=f"X: {int(vec_space[0].item())} mm, Y: {int(vec_space[1].item())} mm, Z: {int(vec_space[2].item())} mm",
                    position=(
                        x1 / img_frame.getWidth() + 0.02,
                        y1 / img_frame.getHeight() + 0.05,
                    ),
                    size=10,
                )
            try:
                label = self._label_map[t.label]
            except Exception:
                label = t.label

            annotation_helper.draw_text(
                text=f"ID: {t.id}, {label}, {t.status.name}",
                position=(
                    x1 / img_frame.getWidth() + 0.02,
                    y1 / img_frame.getHeight() + 0.15,
                ),
                size=10,
                color=SECONDARY_COLOR,
            )

            annotation_helper.draw_rectangle(
                top_left=(x1 / img_frame.getWidth(), y1 / img_frame.getHeight()),
                bottom_right=(x2 / img_frame.getWidth(), y2 / img_frame.getHeight()),
                thickness=2,
                outline_color=SECONDARY_COLOR,
            )

            annotation_helper.draw_text(
                text=f"X: {int(x_space)} mm, Y: {int(y_space)} mm, Z: {int(z_space)} mm",
                position=(
                    x1 / img_frame.getWidth() + 0.02,
                    y1 / img_frame.getHeight() + 0.1,
                ),
                size=10,
                color=SECONDARY_COLOR,
            )

        annotations = annotation_helper.build(
            timestamp=tracklets.getTimestamp(), sequence_num=tracklets.getSequenceNum()
        )
        self.out.send(annotations)

def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-fps",
        "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False,
        default=None,
        type=int,
    )

    args = parser.parse_args()

    return parser, args

_, args = initialize_argparser()


def main():

    CUP_LABEL = 41

    _, args = initialize_argparser()

    visualizer = dai.RemoteConnection(httpPort=8558)
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform_obj = device.getPlatform()
    platform = platform_obj.name
    print(f"Platform: {platform}")

    if args.fps_limit is None:
        args.fps_limit = 20 if platform == "RVC2" else 30

    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        # Detection  model
        # Get model from zoo with platform specification for correct compilation
        #model_desc = dai.NNModelDescription.fromYamlFile("./models/yolo11.yaml") # Doesnt work...
        model_desc = dai.NNModelDescription("luxonis/yolov6-nano:r2-coco-512x288") # Works
        model_desc.platform = platform  # platform is a string (e.g., "RVC2")
        #nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc))
        #labels = nn_archive.getConfig().model.heads[0].metadata.classes
        #cup_label = labels.index("cup") if "cup" in labels else 41

        # Camera input
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=args.fps_limit)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=args.fps_limit)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_cam.requestOutput((640,400)),
            right=right_cam.requestOutput((640,400)),
            presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
        )
        #if platform == "RVC2":
        #    stereo.setOutputSize(*nn_archive.getInputSize())
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        # Build network with archive for RVC2, or model_desc for other platforms
        

        #nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        #    cam, stereo, model_desc, fps=args.fps_limit
        #)
        nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam, stereo, model_desc, fps=args.fps_limit)
        #nn.setNumShavesPerInferenceThread(4)
        #nn.setBlobPath("models/yolo11n_512_288_openvino_2022.1_4shave.blob")
        
        #n.build(input=cam, stereo=stereo, model=model_desc, fps=args.fps_limit)
        #nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc))
        #nn.setNNArchive(nn_archive, numShaves=4)
        
        nn.setBoundingBoxScaleFactor(0.7)
        nn.setDepthLowerThreshold(100)
        nn.setDepthUpperThreshold(5000)

        # Tracking
        object_tracker = pipeline.create(dai.node.ObjectTracker)
        object_tracker.setDetectionLabelsToTrack([CUP_LABEL])
        if platform == "RVC2":
            object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        else:
            object_tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

        nn.passthrough.link(object_tracker.inputTrackerFrame)
        nn.passthrough.link(object_tracker.inputDetectionFrame)
        nn.out.link(object_tracker.inputDetections)

        calibration_handler = device.readCalibration()
        baseline = calibration_handler.getBaselineDistance()*10
        focal_length = calibration_handler.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 400)[0][0]

        # Kalmar filter
        kalmar_filter_node = pipeline.create(KalmanFilterNode).build(
            rgb=nn.passthrough,
            tracker_out=object_tracker.out,
            baseline=baseline,
            focal_length=focal_length,
            label_map={CUP_LABEL: "cup"}
        )
        
        # Visualize
        visualizer.addTopic("Video", nn.passthrough, "images")
        visualizer.addTopic("Tracklets", kalmar_filter_node.out, "images")

        print("Pipeline created successfully")
        
        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            if key == ord('q'):
                break


if __name__ == "__main__":
    main()
