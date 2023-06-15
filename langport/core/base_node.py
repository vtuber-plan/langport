import inspect
import logging

from typing import Any, Callable, Dict, Iterable, Mapping, Optional
from langport.utils.interval_timer import IntervalTimer


class BaseNode(object):
    def __init__(
        self,
        node_addr: str,
        node_id: str,
        logger: logging.Logger,
    ):
        self.node_addr = node_addr
        self.node_id = node_id
        self.logger = logger
        self.online = False

        self.start_fn: Dict[str, Callable[[], None]] = {}
        self.stop_fn: Dict[str, Callable[[], None]] = {}

        self.timers: Dict[str, IntervalTimer] = {}

        self.on_stop("stop_all_timers", self.stop_all_timers)

    async def start(self):
        if self.online:
            return
        for name, fn in self.start_fn.items():
            if inspect.iscoroutinefunction(fn):
                await fn()
            else:
                fn()
        self.online = True

    async def stop(self):
        if not self.online:
            return
        for name, fn in self.stop_fn.items():
            if inspect.iscoroutinefunction(fn):
                await fn()
            else:
                fn()
        self.online = False

    def on_start(self, name: str, fn: Callable[[], None]):
        self.start_fn[name] = fn

    def on_stop(self, name: str, fn: Callable[[], None]):
        self.stop_fn[name] = fn

    def add_timer(
        self,
        name: str,
        interval: float,
        fn: Callable,
        args: Optional[Iterable[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        workers: int = 4,
    ) -> bool:
        if name in self.timers:
            return False
        new_timer = IntervalTimer(
            interval=interval, fn=fn, max_workers=workers, args=args, kwargs=kwargs
        )
        self.timers[name] = new_timer
        new_timer.start()
        return True

    def remove_timer(self, name: str) -> bool:
        if name not in self.timers:
            return False
        self.timers[name].cancel()
        del self.timers[name]
        return True

    def stop_all_timers(self):
        for name, timer in self.timers.items():
            timer.cancel()
        self.timers.clear()
