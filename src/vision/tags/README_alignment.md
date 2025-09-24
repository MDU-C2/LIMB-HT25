# Robot Arm Tag Alignment Detection

This module provides functionality to detect when ArUco tags placed on a robot arm are aligned with each other.

## Robot Arm Tag Configuration

The system is designed for a robot arm with 6 ArUco tags placed as follows:

1. **Tag 0**: Top of the hand
2. **Tag 1**: Side of the hand/on thumb  
3. **Tag 2**: Top of the forearm
4. **Tag 3**: Bottom side of the forearm
5. **Tag 4**: Top of the upper arm (bicep)
6. **Tag 5**: Bottom of the upper arm (tricep)

## Expected Alignments

The system detects the following types of alignments:

### Vertical Alignments
- Hand top ↔ Forearm top (Tags 0-2)
- Hand side ↔ Forearm bottom (Tags 1-3)
- Forearm top ↔ Upper arm top (Tags 2-4)
- Forearm bottom ↔ Upper arm bottom (Tags 3-5)

### Horizontal Alignments
- Hand top ↔ Hand side (Tags 0-1)
- Forearm top ↔ Forearm bottom (Tags 2-3)
- Upper arm top ↔ Upper arm bottom (Tags 4-5)

### Parallel Alignments
- Hand top ↔ Upper arm top (Tags 0-4)
- Hand side ↔ Upper arm bottom (Tags 1-5)

## Usage

The alignment detection is automatically integrated into the tag visualization system. When you run the vision system in tag mode:

```bash
python main.py --mode tag --show
```

The system will:

1. **Detect ArUco tags** in the camera feed
2. **Calculate alignments** between detected tags
3. **Display visual indicators**:
   - Green lines for vertical alignments
   - Blue lines for horizontal alignments  
   - Yellow lines for parallel alignments
4. **Show alignment status** in the top-left corner
5. **Print alignment info** to the terminal

## Visual Indicators

- **Colored lines** connect aligned tags
- **Status text** shows number of detected alignments
- **Detailed info** lists specific alignments with confidence scores
- **Terminal output** provides real-time alignment feedback

## Configuration

You can adjust alignment sensitivity by modifying the `TagAlignmentDetector` parameters:

- `alignment_threshold_deg`: Maximum angle deviation for alignment (default: 15°)
- `distance_threshold_m`: Maximum distance for considering tags as aligned (default: 0.5m)

## Example Output

```
Detected 3 alignments: hand_top ↔ forearm_top (vertical)(0.95) hand_side ↔ forearm_bottom (vertical)(0.87) hand_top ↔ hand_side (horizontal)(0.92)
```

This indicates that the robot arm is in a position where:
- The hand top and forearm top are vertically aligned (95% confidence)
- The hand side and forearm bottom are vertically aligned (87% confidence)  
- The hand top and hand side are horizontally aligned (92% confidence)
