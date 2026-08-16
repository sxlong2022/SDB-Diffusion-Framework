import torch
import gc
import logging

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU Memory Manager"""
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory"""
        try:
            if torch.cuda.is_available():
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                logger.info("GPU memory cleanup completed")
                
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {str(e)}")
            raise
            
    @staticmethod
    def get_gpu_memory_info():
        """Get GPU memory usage information"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                cached_memory = torch.cuda.memory_reserved(device)
                
                return {
                    'total': total_memory / 1024**2,  # MB
                    'allocated': allocated_memory / 1024**2,
                    'cached': cached_memory / 1024**2,
                    'free': (total_memory - allocated_memory) / 1024**2
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {str(e)}")
            raise
