import torch
import gc
import logging

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU内存管理器"""
    
    @staticmethod
    def clear_gpu_memory():
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                # 清理PyTorch缓存
                torch.cuda.empty_cache()
                
                # 强制垃圾回收
                gc.collect()
                
                logger.info("GPU内存清理完成")
                
        except Exception as e:
            logger.error(f"GPU内存清理失败: {str(e)}")
            raise
            
    @staticmethod
    def get_gpu_memory_info():
        """获取GPU内存使用信息"""
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
            logger.error(f"获取GPU内存信息失败: {str(e)}")
            raise
