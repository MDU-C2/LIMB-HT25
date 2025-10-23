# OAK-D Vision System with SpatialVisualizer

## Overview

This vision system uses the **SpatialVisualizer** - a custom DepthAI HostNode that displays:
- **RGB video** with cup detections and spatial coordinates (X, Y, Z in mm)
- **Depth map** with color-coded visualization and bounding boxes
- **AprilTag detection** with pose estimation and 3D coordinates

## Quick Start

### 1. Install Dependencies

```bash
cd src/vision
pip install -r requirements.txt
```

### 2. Run the System

```bash
python main.py
```

Or directly:

```bash
python system.py
```

### 3. Run Extrinsic Calibration (Optional)

To calibrate the camera-to-world transformation using AprilTags:

```bash
python extrinsic_calibration.py
```

Place AprilTag ID 0 (10 cm size) where you want the world origin. Press 's' to save calibration.

### 4. View Results

Three OpenCV windows will appear:
- **RGB** - Color video with cup detections, confidence scores, and spatial coordinates (X, Y, Z in mm)
- **Depth** - Color-coded depth map with bounding boxes
- **AprilTags** - Grayscale view with detected AprilTags, IDs, and poses

Press `q` in any window to quit.

## How It Works

The vision system uses **DepthAI v3's blocking pipeline** with `pipeline.run()`, which:
1. Starts the pipeline
2. Processes frames in the `SpatialVisualizer` HostNode
3. Blocks until the pipeline is stopped (when you press 'q')

### SpatialVisualizer HostNode

The `SpatialVisualizer` is a **HostNode** that runs on your computer and:

1. **Receives three inputs** from the pipeline:
   - `passthroughDepth` - Depth frames from stereo
   - `out` - Spatial detections with 3D coordinates
   - `passthrough` - RGB preview frames

2. **Filters detections** - Only shows cups:
   ```python
   cup_detections = [d for d in detections.detections if d.labelName == "cup"]
   ```

3. **Processes** each frame:
   - Applies hot colormap to depth for visualization
   - Draws bounding boxes on both RGB and depth
   - Overlays detection labels, confidence, and spatial coordinates (X, Y, Z)

4. **Displays** results using OpenCV windows in real-time

5. **Handles quit** - Calls `stopPipeline()` when 'q' is pressed

### Pipeline Architecture

```
Camera RGB ──┬──> Spatial Detection Network ──> Detections (all objects)
             │                                        │
             │   ┌──> ImageManip (GRAY8) ──> AprilTag Detector ──> Tags
             │   │                                                    │
Stereo Depth ┴───┼────────────────────────────────────────┬─────────┤
             │   │                                         │         │
             └───┴───────> SpatialVisualizer ◄────────────┴─────────┘
                           ├─> Filter for cups only
                           ├─> Display RGB window
                           ├─> Display Depth window
                           └─> Display AprilTags window
```

## Key Features

### RGB Window
- Bounding boxes around detected cups
- Label: "cup"
- Confidence score (%)
- **Spatial coordinates (X, Y, Z in millimeters)**

Example display:
```
cup
85.23
X: 120 mm
Y: -45 mm
Z: 1523 mm
```

### Depth Window
- Color-coded depth visualization (hot colormap: dark=close, bright=far)
- Bounding boxes showing detection regions
- Auto-scaled based on min/max depth in scene (uses 1st and 99th percentiles)

### AprilTags Window
- Grayscale preprocessed image
- Detected tags with green outlines
- Tag ID labels at center
- **3D coordinate axes** overlaid on each tag:
  - **Red axis (X)**: Points to the right edge of the tag
  - **Green axis (Y)**: Points to the bottom edge of the tag
  - **Blue axis (Z)**: Points out from the tag (toward camera)
- Axes are 5cm long by default
- Tag count displayed at top

## Code Structure

### system.py

**SpatialVisualizer Class** (lines 15-73)
- `build(depth, detections, rgb)` - Links input streams
- `process(depth, detections, rgb)` - Called for each frame
- `process_depth_frame(depth)` - Normalizes and colorizes depth
- `display_results(rgb, depth, detections)` - Shows windows
- `draw_bounding_box(depth, detection)` - Draws on depth map
- `draw_detections(rgb, detection, w, h)` - Draws on RGB with coords

**VisionSystem Class** (lines 75+)
- `__init__()` - Initialize with parameters
- `_create_and_run_pipeline()` - Create pipeline and run (blocking)
- `start_pipeline()` - Entry point to start the system
- `is_pipeline_running()` - Check if pipeline is running
- `shutdown()` - Clean up resources

### main.py

Simple entry point that:
1. Creates VisionSystem instance
2. Calls `start_pipeline()` which blocks
3. Handles Ctrl+C for graceful shutdown

### extrinsic_calibration.py

**ExtrinsicCalibrator Class** - Performs camera-to-world calibration:
- `create_pipeline()` - Sets up AprilTag detection with optimized settings
- `get_camera_intrinsics()` - Extracts camera matrix from OAK-D calibration
- `estimate_tag_pose()` - Computes 6-DOF pose using cv2.solvePnP
- `compute_transformation_matrix()` - Converts rvec/tvec to 4x4 matrix
- `process_frame()` - Detects reference tag and updates calibration
- `save_calibration()` - Exports T_world_camera to JSON
- `run()` - Main calibration loop with visualization

**Controls:**
- `s` - Save calibration to `extrinsic_calibration.json`
- `r` - Reset calibration samples
- `q` - Quit calibration

## Configuration

### Detection Parameters

Edit in `VisionSystem.__init__()`:

```python
vision_system = VisionSystem(
    model_path=None,                # Use default YOLOv6-nano
    confidence_threshold=0.5,       # Min confidence (0-1)
    spatial_threshold=5000,         # Max depth in mm
    apriltag_family="TAG36H11",     # TAG36H11, TAG25H9, or TAG16H5
    apriltag_quad_decimate=1.5,     # Lower = better quality, slower
    apriltag_quad_sigma=1.0,        # Gaussian blur for noise reduction
    apriltag_refine_edges=True,     # Refine edge detection
    apriltag_max_hamming=1,         # Max bit errors (0-2, lower = stricter)
)
```

### Pipeline Settings

In `_create_and_run_pipeline()`:

```python
FPS = 30                          # Frame rate

# Spatial detection network
spatial_detection_network.setBoundingBoxScaleFactor(0.5)  # ROI for depth
spatial_detection_network.setDepthLowerThreshold(100)     # Min depth (mm)
spatial_detection_network.setDepthUpperThreshold(5000)    # Max depth (mm)

# Stereo depth
stereo.setExtendedDisparity(True)  # Extended range
```

### Camera Resolution

```python
mono_left.requestOutput((640, 400)).link(stereo.left)
mono_right.requestOutput((640, 400)).link(stereo.right)
```

## Detection Filtering

Currently filters to show **only cups**:

```python
cup_detections = [d for d in detections.detections if d.labelName == "cup"]
```

To show **all detections**, change line 28 in `system.py`:

```python
# Show all detections
self.display_results(rgb_preview, depth_frame_color, detections.detections)

# Or filter for specific classes
bottle_detections = [d for d in detections.detections if d.labelName == "bottle"]
person_detections = [d for d in detections.detections if d.labelName == "person"]
```

### Available Classes (YOLOv6 COCO)

The default model detects 80 classes including:
- person, bicycle, car, motorcycle, airplane, bus, train, truck
- bottle, wine glass, **cup**, fork, knife, spoon, bowl
- chair, couch, bed, dining table, toilet
- tv, laptop, mouse, remote, keyboard, cell phone
- And many more...

## Understanding pipeline.run()

The system uses **blocking execution**:

```python
with dai.Pipeline(self.device) as pipeline:
    # Create and configure nodes...
    visualizer.build(...)
    
    self.pipeline = pipeline
    self.pipeline.run()  # ← Blocks here until pipeline stops
```

**Key Points**:
- `pipeline.run()` blocks until `stopPipeline()` is called (from SpatialVisualizer when 'q' is pressed)
- All visualization happens in the HostNode's `process()` method
- Main thread is blocked, so you don't need a `while` loop
- Pipeline automatically cleans up when exiting the `with` block

This is different from `pipeline.start()` which runs asynchronously.

## Customization

### Change Depth Colormap

In `process_depth_frame()` (line 39):

```python
return cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

# Other options:
# cv2.COLORMAP_JET     # Blue (close) to red (far)
# cv2.COLORMAP_TURBO   # Smooth rainbow
# cv2.COLORMAP_VIRIDIS # Perceptually uniform
# cv2.COLORMAP_PLASMA  # Purple to yellow
```

### Change Bounding Box Colors

In `draw_detections()` (line 67):

```python
color = (255, 255, 255)  # White (BGR format)

# Other options:
# (0, 255, 0)    # Green
# (255, 0, 0)    # Blue
# (0, 255, 255)  # Yellow
# (0, 165, 255)  # Orange
```

### Adjust Text Size

In `draw_detections()` (lines 68-72):

```python
cv2.putText(frame, str(label), (x1+10,y1+20), 
            cv2.FONT_HERSHEY_TRIPLEX, 
            0.5,  # ← Font scale (0.5=small, 1.0=large)
            color)
```

### Change Confidence Display

Currently shows as percentage (line 69):

```python
cv2.putText(frame, "{:.2f}".format(detection.confidence*100), ...)

# Show as decimal instead:
cv2.putText(frame, f"{detection.confidence:.2f}", ...)
```

## Platform-Specific Optimizations

The code automatically detects the platform and adjusts:

```python
platform = pipeline.getDefaultDevice().getPlatform()
if platform == dai.Platform.RVC2:
    stereo.setOutputSize(640, 400)
```

**RVC2** devices (OAK-D, OAK-D-Lite) need explicit stereo output size.
**RVC3/RVC4** devices handle this automatically.

## Troubleshooting

### No Windows Appear

**Check if running headless:**
```bash
echo $DISPLAY  # Should show something like ":0" or ":1"
```

**If empty, you're running without a display.** Solutions:
- Run locally on a machine with a display
- Use X11 forwarding: `ssh -X user@host`
- Use VNC or remote desktop

**Test OpenCV:**
```python
import cv2
import numpy as np
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

If no window appears, OpenCV is not configured for GUI.

### Device Not Found

```bash
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

Should show at least one device. If not:
- Use USB 3.0 port
- Try different USB port
- Check cable
- Check permissions (Linux): Add udev rules

### No Detections

- Ensure cups are within 0.1-5 meters
- Improve lighting
- Lower confidence threshold (e.g., 0.3)
- Check that object is actually a cup (model may be specific)

### Poor Depth Quality

- Clean stereo cameras
- Ensure good lighting
- Avoid textureless or reflective surfaces
- Use active IR if available

### Debug Mode

Add prints to `SpatialVisualizer.process()`:

```python
def process(self, depth_preview, detections, rbg_preview):
    print(f"Process called! Detections: {len(detections.detections)}")
    # ... rest of code
```

## Performance

Typical performance:
- **FPS**: 25-30 on most systems
- **Latency**: ~35-50ms end-to-end
- **Depth Range**: 0.1 - 5 meters (configurable)
- **CPU Usage**: ~40-60% (mainly for visualization)

## Integration Example

Use in your own code:

```python
from system import VisionSystem

# Create vision system
vision = VisionSystem(
    confidence_threshold=0.6,
    spatial_threshold=3000  # 3 meters
)

# This will block until user presses 'q' in window
vision.start_pipeline()

# Cleanup (if needed)
vision.shutdown()
```

The pipeline runs and blocks in `start_pipeline()`, with all visualization handled by the HostNode.

## Model Information

**Default Model**: YOLOv6-Nano
- Fast and efficient
- Trained on COCO dataset (80 classes)
- Optimized for OAK-D devices
- Downloaded automatically from DepthAI model zoo

To use a different model:
```python
model_desc = dai.NNModelDescription("your-model-name")
```

## AprilTag Detection Configuration

### Tuning Detection Parameters

The AprilTag detector has been optimized with preprocessing and configurable thresholds:

**Preprocessing Pipeline:**
- Converts RGB to grayscale (`GRAY8`) before detection
- Reduces computational load and improves contrast
- 640×400 resolution for main system, 1280×800 for calibration

**Detector Thresholds:**

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `quad_decimate` | 1.5 | Image decimation factor | Lower (1.0) = better small tag detection, slower<br>Higher (2.0-3.0) = faster, miss small tags |
| `quad_sigma` | 1.0 | Gaussian blur sigma | 0.0 = no blur (clean images)<br>0.8-1.2 = reduce noise (grainy images) |
| `refine_edges` | True | Subpixel edge refinement | True = better pose accuracy<br>False = faster detection |
| `max_hamming` | 1 | Max bit errors allowed | 0 = no errors, fewer false positives<br>1-2 = more tolerant, detects damaged tags |

**Quick Optimization Guide:**

For **better accuracy** (slower):
```python
apriltag_quad_decimate=1.0,
apriltag_quad_sigma=0.8,
apriltag_refine_edges=True,
apriltag_max_hamming=0,
```

For **better speed** (less accurate):
```python
apriltag_quad_decimate=2.0,
apriltag_quad_sigma=0.0,
apriltag_refine_edges=False,
apriltag_max_hamming=1,
```

Balanced (recommended):
```python
apriltag_quad_decimate=1.5,
apriltag_quad_sigma=1.0,
apriltag_refine_edges=True,
apriltag_max_hamming=1,
```

## Extrinsic Calibration Guide

### Overview

Extrinsic calibration establishes the transformation from camera coordinates to world/robot coordinates using AprilTags as reference markers.

### Requirements

1. **AprilTag**: Print TAG36H11 family, ID 0, at **10 cm** size
   - Use high-contrast matte paper
   - Ensure flat mounting (no warping)
   - Place at desired world origin location

2. **Lighting**: Even, diffuse lighting without harsh shadows or glare

3. **Setup**: Mount tag rigidly where you want (0, 0, 0) to be

### Calibration Process

1. **Run calibration script:**
   ```bash
   cd src/vision
   python extrinsic_calibration.py
   ```

2. **Position camera**: Point camera at reference tag (ID 0)

3. **Collect samples**: Move camera to different angles (30+ samples recommended)
   - Vary viewing angles (±30° from frontal)
   - Maintain 0.5-2 meters distance
   - Keep tag fully visible

4. **Monitor quality**: Watch "Samples: X/30" counter and ensure tag is detected consistently

5. **Save calibration**: Press `s` when satisfied with sample count

6. **Output**: Creates `extrinsic_calibration.json` with:
   - `T_world_camera`: 4×4 transformation matrix (camera pose in world frame)
   - `camera_matrix`: 3×3 intrinsic calibration
   - `dist_coeffs`: Distortion coefficients
   - Metadata (tag size, family, timestamp)

### Using Calibration Results

Load and apply calibration in your code:

```python
import json
import numpy as np

# Load calibration
with open('extrinsic_calibration.json', 'r') as f:
    calib = json.load(f)

T_world_camera = np.array(calib['T_world_camera'])
camera_matrix = np.array(calib['camera_matrix'])

# Transform point from camera to world
point_camera = np.array([x, y, z, 1])  # Homogeneous coordinates
point_world = T_world_camera @ point_camera

print(f"World coordinates: {point_world[:3]}")
```

### Coordinate Frames

**Camera Frame** (before calibration):
- Origin: Camera optical center
- X: Right, Y: Down, Z: Forward

**World Frame** (after calibration):
- Origin: Reference tag center (ID 0)
- X: Tag right edge, Y: Tag bottom edge, Z: Out of tag (toward camera)

**Transformation**: `T_world_camera` converts camera coordinates → world coordinates

### Troubleshooting Calibration

| Issue | Solution |
|-------|----------|
| Tag not detected | Check tag size (10 cm), lighting, focus |
| Low sample count | Move camera more, ensure tag stays in view |
| High reprojection error | Improve lighting, check tag flatness, reduce distance |
| Jittery pose | Add more samples, improve lighting, use lower `quad_decimate` |

## Next Steps

Potential enhancements:

1. **Multi-Tag Calibration**
   - Use multiple tags for redundancy
   - Average transformations for robustness

2. **Record Detections**
   - Log detection data to file
   - Save video with annotations

3. **Multiple Object Classes**
   - Detect bottles, bowls, etc.
   - Different colors per class

4. **Hand-Eye Calibration**
   - Compute robot end-effector to camera transformation
   - Enable robot-guided manipulation

5. **Dynamic Calibration**
   - Continuous stereo recalibration during operation
   - Compensate for thermal drift and mechanical changes

6. **Remote Viewing**
   - Add web interface
   - Stream to remote display

## Architecture Notes

### Why HostNode?

HostNodes allow custom Python processing on the host computer (not the OAK-D device):
- Full access to Python libraries (NumPy, OpenCV)
- Easy debugging and visualization
- Flexible processing
- Direct display with cv2.imshow()

### Why pipeline.run()?

`pipeline.run()` provides a simpler execution model:
- Automatically blocks until completion
- No need for manual queue polling
- Automatic cleanup on exit
- Clean integration with `with` statement

This is ideal for applications where visualization is the primary goal.

### Alternative: Async with start()

For non-blocking operation, use `pipeline.start()`:

```python
pipeline.start()
while pipeline.isRunning():
    # Do other work
    time.sleep(0.1)
```

## Coordinate System

- **Origin**: RGB camera optical center
- **X-axis**: Right (positive to the right)
- **Y-axis**: Down (positive downward)
- **Z-axis**: Forward (positive away from camera)
- **Units**: Millimeters (mm)

Example: `X: 120 mm, Y: -45 mm, Z: 1523 mm` means:
- 120mm to the right of camera center
- 45mm above camera center (negative Y)
- 1523mm away from camera

## License

Part of the LIMB-HT25 project.

## Dependencies

- Python 3.10+
- depthai >= 3.0.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0

---

## Summary of Recent Improvements

### AprilTag Detection Enhancements
✅ **Preprocessing**: Grayscale conversion via `ImageManip` for better contrast  
✅ **Configurable Thresholds**: Quad decimation, sigma blur, edge refinement, max Hamming  
✅ **Family Support**: TAG36H11, TAG25H9, TAG16H5 with proper enum mapping  
✅ **Visualization**: Real-time tag detection with IDs and outlines  

### Extrinsic Calibration System
✅ **AprilTag-Based**: Uses reference tag to define world coordinate frame  
✅ **Sample Averaging**: Collects 30 samples for robust calibration  
✅ **Pose Estimation**: 6-DOF pose using `cv2.solvePnP` with IPPE_SQUARE  
✅ **JSON Export**: Saves transformation matrix, intrinsics, and metadata  
✅ **Live Visualization**: Shows 3D axes, tag outlines, and calibration status  

### Configuration Options
✅ **Detection Quality**: Adjustable speed vs. accuracy tradeoff  
✅ **Multiple Tag Families**: Support for different AprilTag standards  
✅ **Flexible Setup**: Works with single tags or boards  

---

**Version**: 2.0  
**Last Updated**: 2025-10-21  
**Status**: Production Ready with AprilTag Support ✅
