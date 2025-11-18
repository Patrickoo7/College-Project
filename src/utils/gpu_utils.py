"""GPU detection and management utilities."""

import os
import platform
from typing import Dict, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


class GPUManager:
    """Manage GPU detection and configuration for ML libraries."""

    def __init__(self):
        """Initialize GPU Manager."""
        self._gpu_available = None
        self._gpu_info = None
        self._cuda_available = None
        self._device_count = 0

    def detect_gpu(self) -> Tuple[bool, Dict]:
        """
        Detect GPU availability and gather information.

        Returns:
            Tuple of (gpu_available, gpu_info_dict)
        """
        if self._gpu_available is not None:
            return self._gpu_available, self._gpu_info

        gpu_info = {
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cuda_version": None,
            "cudnn_version": None,
            "platform": platform.system(),
        }

        # Check CUDA via PyTorch (most reliable)
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["device_count"] = torch.cuda.device_count()
                gpu_info["device_names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(gpu_info["device_count"])
                ]
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["cudnn_version"] = torch.backends.cudnn.version()
                logger.info(f"PyTorch CUDA detected: {gpu_info['device_count']} GPU(s)")
                for i, name in enumerate(gpu_info["device_names"]):
                    logger.info(f"  GPU {i}: {name}")
        except ImportError:
            logger.debug("PyTorch not available for CUDA detection")

        # Check CUDA via TensorFlow if PyTorch not available
        if not gpu_info["cuda_available"]:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                gpu_info["cuda_available"] = len(gpus) > 0
                gpu_info["device_count"] = len(gpus)
                gpu_info["device_names"] = [gpu.name for gpu in gpus]
                if gpu_info["cuda_available"]:
                    logger.info(f"TensorFlow GPU detected: {gpu_info['device_count']} GPU(s)")
            except ImportError:
                logger.debug("TensorFlow not available for GPU detection")

        # Check for Apple Silicon GPU (MPS)
        if platform.system() == "Darwin":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    gpu_info["mps_available"] = True
                    gpu_info["device_count"] = 1
                    gpu_info["device_names"] = ["Apple Silicon GPU (MPS)"]
                    logger.info("Apple Silicon GPU (MPS) detected")
            except (ImportError, AttributeError):
                gpu_info["mps_available"] = False

        self._gpu_available = gpu_info["cuda_available"] or gpu_info.get("mps_available", False)
        self._gpu_info = gpu_info
        self._cuda_available = gpu_info["cuda_available"]
        self._device_count = gpu_info["device_count"]

        if self._gpu_available:
            logger.info("GPU acceleration ENABLED")
        else:
            logger.info("GPU not available. Using CPU.")

        return self._gpu_available, gpu_info

    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available.

        Returns:
            True if GPU is available, False otherwise
        """
        if self._gpu_available is None:
            self.detect_gpu()
        return self._gpu_available

    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            True if CUDA is available, False otherwise
        """
        if self._cuda_available is None:
            self.detect_gpu()
        return self._cuda_available

    def get_device_count(self) -> int:
        """
        Get number of available GPU devices.

        Returns:
            Number of GPU devices
        """
        if self._device_count == 0:
            self.detect_gpu()
        return self._device_count

    def get_gpu_info(self) -> Dict:
        """
        Get detailed GPU information.

        Returns:
            Dictionary with GPU information
        """
        if self._gpu_info is None:
            self.detect_gpu()
        return self._gpu_info

    def configure_xgboost(self, params: Dict) -> Dict:
        """
        Configure XGBoost parameters for GPU if available.

        Args:
            params: Original XGBoost parameters

        Returns:
            Updated parameters with GPU settings
        """
        gpu_params = params.copy()

        if self.is_cuda_available():
            gpu_params["tree_method"] = "gpu_hist"
            gpu_params["predictor"] = "gpu_predictor"
            # Optionally set GPU device
            if "gpu_id" not in gpu_params:
                gpu_params["gpu_id"] = 0
            logger.info("XGBoost configured for GPU acceleration (gpu_hist)")
        else:
            gpu_params["tree_method"] = params.get("tree_method", "hist")
            logger.info("XGBoost using CPU (hist)")

        return gpu_params

    def configure_lightgbm(self, params: Dict) -> Dict:
        """
        Configure LightGBM parameters for GPU if available.

        Args:
            params: Original LightGBM parameters

        Returns:
            Updated parameters with GPU settings
        """
        gpu_params = params.copy()

        if self.is_cuda_available():
            gpu_params["device"] = "gpu"
            gpu_params["gpu_platform_id"] = 0
            gpu_params["gpu_device_id"] = 0
            logger.info("LightGBM configured for GPU acceleration")
        else:
            gpu_params["device"] = "cpu"
            logger.info("LightGBM using CPU")

        return gpu_params

    def configure_catboost(self, params: Dict) -> Dict:
        """
        Configure CatBoost parameters for GPU if available.

        Args:
            params: Original CatBoost parameters

        Returns:
            Updated parameters with GPU settings
        """
        gpu_params = params.copy()

        if self.is_cuda_available():
            gpu_params["task_type"] = "GPU"
            gpu_params["devices"] = "0"  # GPU device ID
            logger.info("CatBoost configured for GPU acceleration")
        else:
            gpu_params["task_type"] = "CPU"
            logger.info("CatBoost using CPU")

        return gpu_params

    def configure_tensorflow(self):
        """Configure TensorFlow for GPU if available."""
        try:
            import tensorflow as tf

            if self.is_cuda_available():
                # Enable memory growth to avoid OOM
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow configured for {len(gpus)} GPU(s)")
            else:
                # Force CPU
                tf.config.set_visible_devices([], 'GPU')
                logger.info("TensorFlow using CPU")

        except ImportError:
            logger.debug("TensorFlow not available")

    def configure_pytorch(self):
        """Configure PyTorch for GPU if available and return device."""
        try:
            import torch

            if self.is_cuda_available():
                device = torch.device("cuda")
                logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
            elif platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("PyTorch using Apple Silicon GPU (MPS)")
                else:
                    device = torch.device("cpu")
                    logger.info("PyTorch using CPU")
            else:
                device = torch.device("cpu")
                logger.info("PyTorch using CPU")

            return device

        except ImportError:
            logger.debug("PyTorch not available")
            return None

    def set_memory_limit(self, limit_mb: Optional[int] = None):
        """
        Set GPU memory limit for TensorFlow.

        Args:
            limit_mb: Memory limit in MB. If None, uses memory growth instead
        """
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if limit_mb:
                    # Set memory limit
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=limit_mb
                        )]
                    )
                    logger.info(f"GPU memory limit set to {limit_mb}MB")
                else:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("GPU memory growth enabled")

        except Exception as e:
            logger.warning(f"Failed to set GPU memory limit: {str(e)}")

    def print_gpu_summary(self):
        """Print a summary of GPU availability and configuration."""
        gpu_available, gpu_info = self.detect_gpu()

        print("\n" + "="*60)
        print("GPU CONFIGURATION SUMMARY")
        print("="*60)

        if gpu_available:
            print("✓ GPU Acceleration: ENABLED")
            print(f"  Platform: {gpu_info['platform']}")
            print(f"  Device Count: {gpu_info['device_count']}")

            if gpu_info.get('cuda_available'):
                print(f"  CUDA Version: {gpu_info.get('cuda_version', 'N/A')}")
                print(f"  cuDNN Version: {gpu_info.get('cudnn_version', 'N/A')}")

            print("\n  Available Devices:")
            for i, name in enumerate(gpu_info['device_names']):
                print(f"    [{i}] {name}")

            print("\n  Supported Libraries:")
            print("    ✓ XGBoost (gpu_hist)")
            print("    ✓ LightGBM (GPU)")
            print("    ✓ CatBoost (GPU)")

        else:
            print("✗ GPU Acceleration: DISABLED")
            print("  Using CPU for all computations")
            print("\n  To enable GPU acceleration:")
            print("    1. Install CUDA Toolkit")
            print("    2. Install GPU versions of libraries:")
            print("       pip install xgboost[gpu] lightgbm[gpu] catboost[gpu]")

        print("="*60 + "\n")


# Global GPU manager instance
_gpu_manager_instance: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """
    Get or create global GPU manager instance (singleton pattern).

    Returns:
        Global GPUManager instance
    """
    global _gpu_manager_instance
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUManager()
    return _gpu_manager_instance


def is_gpu_available() -> bool:
    """
    Quick check if GPU is available.

    Returns:
        True if GPU is available, False otherwise
    """
    return get_gpu_manager().is_gpu_available()


def configure_gpu_environment():
    """
    Configure GPU environment variables for optimal performance.

    This sets various environment variables that can improve GPU utilization.
    """
    gpu_manager = get_gpu_manager()

    if gpu_manager.is_cuda_available():
        # XGBoost GPU settings
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default

        # TensorFlow GPU settings
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

        logger.info("GPU environment variables configured")
