
import time
from typing import Any

class CacheState(object):
    def __init__(self, value: Any, ttl: int) -> None:
        self.value = value
        self.ttl = ttl
        self.last_time = time.time()
    
    def set(self, value: Any, ttl: int):
        self.value = value
        self.ttl = ttl
        self.last_time = time.time()
    
    def is_valid(self) -> bool:
        now_time = time.time()
        return now_time - self.last_time < self.ttl
    
    def get(self):
        return self.value