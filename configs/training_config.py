import logging

class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Model settings
    MODEL_PATH = "yolo11n.pt"  # Switching to a medium model for better balance
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/model_training/new_dataset/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 100  
    IMAGE_SIZE = 640
    BATCH_SIZE = 96  # Reduced to avoid OOM errors
    ENABLE_MIXED_PRECISION = True

    # Learning Rate Scheduling
    LR0 = 0.01  # Reduced for stability
    LRF = 0.4  
    MOMENTUM = 0.95  
    WEIGHT_DECAY = 0.0001  
    WARMUP_EPOCHS = 5  

    # Gradient Accumulation (adjusted for VRAM)
    GRADIENT_ACCUMULATION_STEPS = 1  

    # Device configuration
    FORCE_CPU = False

    # Logging and Checkpoints
    SAVE_INTERVAL = 5
    LOGGING_INTERVAL = 200
    EARLY_STOP_PATIENCE = 20  

    # Validation Settings
    VAL_INTERVAL = 3  # More frequent validation for debugging

    # Augmentation parameters (fine-tuned)
    AUG_DEGREES = 15.0  
    AUG_TRANSLATE = 0.15  
    AUG_SCALE = 0.6  
    AUG_FLIPLR = 0.5  
    AUG_FLIPUD = 0.2  
    AUG_MOSAIC = 0.8  
    AUG_MIXUP = 0.2  
    AUG_COPY_PASTE = 0.2  
    AUG_HSV_H = 0.02  
    AUG_HSV_S = 0.8  
    AUG_HSV_V = 0.5  

    # Regularization
    DROPOUT_RATE = 0.1  # Reduced for better convergence

    # Add this line in TrainingConfig
    NUM_WORKERS = 0  # Set to 0 for Windows to avoid worker exit errors