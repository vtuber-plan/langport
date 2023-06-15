from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
import time
import asyncio
import traceback
from typing import Any, Iterable, Mapping, Optional


class IntervalTimer(object):
    def __init__(
        self,
        interval: float,
        fn: object,
        max_workers: int = 4,
        args: Optional[Iterable[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
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

    def function_wrapper(
        self,
        args: Optional[Iterable[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ):
        try:
            if inspect.iscoroutinefunction(self.fn):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.fn(*args, **kwargs))
                loop.close()
            else:
                self.fn(*args, **kwargs)
        except:
            traceback.print_exc()

    def run(self):
        while True:
            if self.is_activated:
                start_time = time.time()
                if start_time - self.last_time > self.interval:
                    self._executor.submit(self.function_wrapper, self.args, self.kwargs)
                    self.last_time = time.time()
                    # Never run this code, because finish_time - start_time = 0.
                    # if finish_time - start_time > self.interval:
                    #     print(
                    #         f"Overloaded!! Last timer using {finish_time - start_time}s."
                    #     )
                    # else:
                        # time.sleep(0.01)
                else:
                    time.sleep(0.01)
            else:
                break
