
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any, Iterable, Mapping, Optional

class IntervalTimer(object):
    def __init__(self, interval: float, fn: object, max_workers: int=4, args: Optional[Iterable[Any]]=None, kwargs: Optional[Mapping[str, Any]]=None) -> None:
        self.interval = interval
        self.fn = fn
        self.max_workers = max_workers
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

        self._timer = threading.Thread(target=self.run)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_activated = False
        self.last_time = time.time()
    
    def start(self):
        self.is_activated = True
        self.last_time = time.time()
        self._timer.start()

    def cancel(self):
        self.is_activated = False
        self._timer.join()
    
    def run(self):
        while True:
            if self.is_activated:
                start_time = time.time()
                if start_time - self.last_time > self.interval:
                    self._executor.submit(self.fn, *self.args, **self.kwargs)
                    finish_time = time.time()
                    self.last_time = finish_time
                    if finish_time - start_time > self.interval:
                        print(f"Overloaded!! Last timer using {finish_time - start_time}s.")
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
            else:
                break
            