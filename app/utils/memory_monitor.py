import os
import psutil
import threading
import time
import logging
from app.core.config import settings

logger = logging.getLogger("memory-monitor")


class MemoryMonitor:
    def __init__(self, limit_mb=3500, check_interval=5):
        self.limit_mb = limit_mb
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None

    def start(self):
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started. Limit: {self.limit_mb}MB")

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Memory monitoring stopped")

    def _monitor_memory(self):
        while self.running:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)

                if memory_usage_mb > self.limit_mb:
                    logger.warning(f"Memory usage ({memory_usage_mb:.2f}MB) exceeds limit ({self.limit_mb}MB)")
                    self._free_memory()

                if memory_usage_mb > self.limit_mb * 0.8:
                    logger.info(
                        f"Memory usage: {memory_usage_mb:.2f}MB ({(memory_usage_mb / self.limit_mb) * 100:.1f}% of limit)")

                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)

    def _free_memory(self):

        try:
            import gc
            gc.collect()

            if hasattr(self, 'cache'):
                self.cache.clear()

            logger.info("Memory cleanup performed")
        except Exception as e:
            logger.error(f"Error in memory cleanup: {e}")


memory_monitor = MemoryMonitor(limit_mb=settings.MAX_MEMORY_USAGE_MB)

if settings.ENABLE_MEMORY_MONITORING:
    memory_monitor.start()