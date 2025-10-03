import os
import shutil
from typing import Optional
from pathlib import Path


def ensure_yolo_model_in_models_dir(model_name: str, models_dir: str) -> str:
    """
    Ensure YOLO model exists in the models directory.
    If the model doesn't exist locally, download it to the models directory.
    
    Args:
        model_name: Name of the YOLO model (e.g., 'yolo11s.pt')
        models_dir: Path to the models directory
        
    Returns:
        Full path to the model file in the models directory
    """
    # Ensure model name has .pt extension
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Full path to model in models directory
    model_path = os.path.join(models_dir, model_name)
    
    # If model already exists in models directory, return it
    if os.path.isfile(model_path):
        return model_path
    
    # Model doesn't exist in models directory, need to download it
    print(f"Model {model_name} not found in {models_dir}, downloading...")
    
    try:
        from ultralytics import YOLO
        
        # Change to models directory temporarily to download there
        original_cwd = os.getcwd()
        try:
            os.chdir(models_dir)
            
            # Download the model (this will create it in the current directory)
            model = YOLO(model_name)
            
            # Verify the model was downloaded
            if os.path.isfile(model_name):
                print(f"Successfully downloaded {model_name} to {models_dir}")
                return os.path.join(models_dir, model_name)
            else:
                raise RuntimeError(f"Failed to download {model_name}")
                
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            
    except Exception as e:
        raise RuntimeError(f"Failed to download YOLO model {model_name}: {e}")


def get_yolo_model_path(model_name: str, vision_dir: Optional[str] = None) -> str:
    """
    Get the full path to a YOLO model, ensuring it's in the models directory.
    
    Args:
        model_name: Name of the YOLO model (e.g., 'yolo11s.pt')
        vision_dir: Path to the vision directory (defaults to current file's parent)
        
    Returns:
        Full path to the model file
    """
    if vision_dir is None:
        # Get the directory of this file (utils/)
        utils_dir = os.path.dirname(__file__)
        vision_dir = os.path.dirname(utils_dir)
    
    models_dir = os.path.join(vision_dir, "models")
    return ensure_yolo_model_in_models_dir(model_name, models_dir)


def list_available_models(models_dir: str) -> list:
    """
    List all available YOLO models in the models directory.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        List of model filenames
    """
    if not os.path.isdir(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.pt'):
            models.append(file)
    
    return sorted(models)


def cleanup_downloaded_models(vision_dir: str):
    """
    Clean up any YOLO models that were downloaded to the wrong location.
    This function looks for .pt files in the vision directory and moves them to models/.
    
    Args:
        vision_dir: Path to the vision directory
    """
    models_dir = os.path.join(vision_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Look for .pt files in the vision directory
    moved_count = 0
    for file in os.listdir(vision_dir):
        if file.endswith('.pt'):
            source_path = os.path.join(vision_dir, file)
            dest_path = os.path.join(models_dir, file)
            
            # Move the file to models directory
            try:
                shutil.move(source_path, dest_path)
                print(f"Moved {file} to models directory")
                moved_count += 1
            except Exception as e:
                print(f"Failed to move {file}: {e}")
    
    if moved_count > 0:
        print(f"Moved {moved_count} model file(s) to models directory")
    else:
        print("No model files found to move")


def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path exists and is a valid YOLO model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if the model is valid, False otherwise
    """
    if not os.path.isfile(model_path):
        return False
    
    if not model_path.endswith('.pt'):
        return False
    
    # Try to load the model to validate it
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model is not None
    except Exception:
        return False
