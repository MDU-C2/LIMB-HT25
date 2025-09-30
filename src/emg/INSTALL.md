# EMG Package Installation Guide

This guide provides instructions for installing the EMG signal processing package and its dependencies.

## Quick Installation

### Option 1: Install with pip (Recommended)

```bash
# Navigate to the EMG package directory
cd src/emg

# Install with all dependencies
pip install -r requirements.txt
```

### Option 2: Install as a package

```bash
# Navigate to the EMG package directory
cd src/emg

# Install in development mode
pip install -e .

# Or install with specific extras
pip install -e .[visualization,jupyter]
```

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv emg_env
   source emg_env/bin/activate  # On Windows: emg_env\Scripts\activate
   ```

2. **Install PyTorch**
   ```bash
   # For CPU only
   pip install torch torchvision

   # For GPU support (CUDA)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install core dependencies**
   ```bash
   pip install numpy scipy scikit-learn pywavelets
   ```

4. **Install optional dependencies**
   ```bash
   # For visualization
   pip install matplotlib seaborn

   # For data handling
   pip install pandas

   # For Jupyter notebooks
   pip install jupyter ipywidgets

   # For advanced signal processing
   pip install librosa sounddevice
   ```

## Verification

Test the installation by running the test script:

```bash
cd src
python test_emg_package.py
```

Or run the example usage:

```bash
cd src
python -m emg.example_usage
```

## Troubleshooting

### Common Issues

1. **PyTorch Installation Issues**
   - Visit [PyTorch website](https://pytorch.org/get-started/locally/) for platform-specific instructions
   - Ensure you have the correct CUDA version if using GPU

2. **PyWavelets Installation Issues**
   ```bash
   # If pip fails, try conda
   conda install pywavelets
   ```

3. **Import Errors**
   - Ensure you're in the correct directory (`src/`)
   - Check that all dependencies are installed: `pip list`

4. **Memory Issues**
   - Reduce batch size in training
   - Use smaller model dimensions
   - Process data in smaller chunks

### Platform-Specific Notes

**Windows:**
- Use `python -m pip` instead of `pip` if you encounter issues
- Install Visual C++ Build Tools if compilation fails

**macOS:**
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies if needed

**Linux:**
- Install build essentials: `sudo apt-get install build-essential`
- Install Python development headers: `sudo apt-get install python3-dev`

## Development Installation

For development work, install in editable mode with development dependencies:

```bash
cd src/emg
pip install -e .[dev]
```

This includes:
- pytest for testing
- black for code formatting
- flake8 for linting

## Docker Installation (Optional)

Create a Dockerfile for containerized installation:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "emg.example_usage"]
```

Build and run:
```bash
docker build -t emg-package .
docker run -it emg-package
```

## Performance Optimization

### GPU Acceleration

For faster training and inference:

1. **Install CUDA-enabled PyTorch**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Verify GPU availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   ```

### CPU Optimization

For CPU-only systems:

1. **Install optimized BLAS libraries**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libopenblas-dev

   # On macOS with Homebrew
   brew install openblas
   ```

2. **Set environment variables**
   ```bash
   export OPENBLAS_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

## Uninstallation

To remove the package:

```bash
pip uninstall emg-signal-processing
```

Or if installed in development mode:

```bash
pip uninstall -e .
```
