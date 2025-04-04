import logging

class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Model settings
    MODEL_PATH = "yolo11x.pt"  # Using a larger model for better accuracy
    DATA_YAML_PATH = "D:/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 200  # Increased from 300 for better convergence
    IMAGE_SIZE = 640
    BATCH_SIZE = 112  # Increased from 64 to improve generalization
    ENABLE_MIXED_PRECISION = True

    # Learning Rate Scheduling
    LR0 = 0.02  # Increased for faster initial learning
    LRF = 0.4  # Adjusted for better learning rate decay
    MOMENTUM = 0.95  # Increased for more stable training
    WEIGHT_DECAY = 0.0001  # Reduced to prevent over-regularization
    WARMUP_EPOCHS = 5  # Increased for better early training stability

    # Gradient Accumulation (for larger effective batch size if needed)
    GRADIENT_ACCUMULATION_STEPS = 2  # Helps when VRAM is insufficient
    
    # Device configuration
    FORCE_CPU = False

    # Logging and Checkpoints
    SAVE_INTERVAL = 5
    LOGGING_INTERVAL = 200
    EARLY_STOP_PATIENCE = 20  # Stops training if no improvement in 20 epochs

    # Validation Settings
    VAL_INTERVAL = 5  # Validate every 5 epochs instead of 1 to save time

    # Augmentation parameters (enhanced for better generalization)
    AUG_DEGREES = 15.0  # Slightly increased rotation range
    AUG_TRANSLATE = 0.15  # Increased translation range
    AUG_SCALE = 0.6  # Increased scale range
    AUG_FLIPLR = 0.5  # Horizontal flip probability remains the same
    AUG_FLIPUD = 0.2  # Added vertical flip probability for better variety
    AUG_MOSAIC = 0.8  # Reduced mosaic probability slightly to prevent artifacts
    AUG_MIXUP = 0.2  # Increased Mixup augmentation probability
    AUG_COPY_PASTE = 0.2  # Increased Copy-Paste augmentation
    AUG_HSV_H = 0.02  # Slightly increased hue augmentation
    AUG_HSV_S = 0.8  # Increased saturation augmentation
    AUG_HSV_V = 0.5  # Increased value augmentation for brightness variations

    # Regularization
    DROPOUT_RATE = 0.2  # Added dropout to prevent overfitting