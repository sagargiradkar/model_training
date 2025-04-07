import torch
import logging
import gc
from config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

class DeviceManager:
    @staticmethod
    def get_device():
        """Returns the appropriate device (CPU or CUDA) based on availability and settings."""
        if TrainingConfig.FORCE_CPU:
            logger.info("FORCE_CPU is enabled. Using CPU.")
            return torch.device('cpu')

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {device_name} (Total GPUs available: {device_count})")
            return torch.device('cuda:0')  # Change 'cuda:0' if you want to select a different GPU
        else:
            logger.info("CUDA not available. Using CPU.")
            return torch.device('cpu')

    @staticmethod
    def clear_cuda_memory():
        """Clears unused CUDA memory and forces garbage collection."""
        if torch.cuda.is_available():
            logger.info("Clearing CUDA memory...")
            torch.cuda.empty_cache()
            gc.collect()  # Helps clear memory occupied by deleted objects
            logger.info(f"CUDA cache cleared. Free memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    @staticmethod
    def log_gpu_memory():
        """Logs GPU memory usage for debugging potential memory leaks."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9  # Convert bytes to GB
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
