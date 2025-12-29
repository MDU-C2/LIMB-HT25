# Configuration System

The LIMB-HT25 system uses YAML configuration files to initialize all layers and hardware interfaces.

## Configuration File

The main configuration file is `system_config.yaml` located in `src/config/`. This file contains all parameters for:

- **Hardware interfaces** (CAN, BLE)
- **Input layer** (window size, sample rate)
- **Processing layer** (EMG processing, ML models, IMU, fusion)
- **Control layer** (control rate, thresholds, workspace limits)
- **Vision system** (detection thresholds, AprilTag settings)
- **Queue configuration**

## Usage

### Using the default config file

```bash
python -m layers.main
```

The system will automatically search for `config/system_config.yaml` in standard locations.

### Using a custom config file

```bash
python -m layers.main --config /path/to/custom_config.yaml
```

### Overriding with command-line arguments

You can still override specific parameters via command-line arguments:

```bash
python -m layers.main --config config.yaml --can-interface can1 --control-rate 200.0
```

## Configuration Structure

The YAML file is organized hierarchically:

```yaml
hardware:
  can:
    interface: "can0"
    bitrate: 1000000
  ble:
    device_name: "LIMBServer"
    scan_timeout: 10.0

input_layer:
  window_size: 100
  sample_rate: 100.0

processing_layer:
  model:
    path: null
    scaler_path: null
  emg:
    fs: 1000.0
    lowcut: 20.0
    # ... more parameters

control_layer:
  control_rate: 100.0
  conf_threshold: 0.5
  # ... more parameters

vision:
  confidence_threshold: 0.5
  spatial_threshold: 5000
  # ... more parameters
```

## Benefits

1. **Centralized configuration**: All parameters in one place
2. **Version control**: Easy to track changes and maintain different configs
3. **Comments**: YAML supports comments for documentation
4. **Environment-specific configs**: Use different configs for dev/test/prod
5. **No code changes**: Adjust parameters without modifying code

## Default Values

If a config file is not provided or a parameter is missing, the system uses sensible defaults defined in `layers/main.py`.

## Requirements

- PyYAML (`pip install pyyaml`)

