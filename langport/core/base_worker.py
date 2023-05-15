import argparse
import asyncio
from collections import defaultdict
import dataclasses
import logging
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
import threading
import queue
import uuid

import requests
from tenacity import retry, stop_after_attempt

from langport.protocol.worker_protocol import (
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    WorkerHeartbeatPing,
    WorkerHeartbeatPong,
    WorkerStatus,
)

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    ErrorCode,
)
from langport.utils.interval_timer import IntervalTimer


class BaseWorker(object):
    def __init__(
        self,
        controller_addr: Optional[str],
        worker_addr: str,
        worker_id: str,
        worker_type: str,
        logger: logging.Logger,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.logger = logger
        self.online = False

        self.start_fn: Dict[str, Callable[[], None]] = {}
        self.stop_fn: Dict[str, Callable[[], None]] = {}

        self.timers: Dict[str, IntervalTimer] = {}

        # startup
        if self.controller_addr is not None:
            self.on_start("register_heatbeat", self.register_heatbeat)
            self.on_start("register_to_controller", self.register_to_controller)

        if self.controller_addr is not None:
            self.on_stop("remove_from_controller", self.remove_from_controller)
        self.on_stop("stop_all_timers", self.stop_all_timers)

    def start(self):
        if self.online:
            return
        for name, fn in self.start_fn.items():
            fn()
        self.online = True

    def stop(self):
        if not self.online:
            return
        for name, fn in self.stop_fn.items():
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
        fn: Callable[["BaseWorker"], None],
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

    def register_to_controller(self):
        self.logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = RegisterWorkerRequest(
            worker_id=self.worker_id,
            worker_addr=self.worker_addr,
            worker_type=self.worker_type,
            check_heart_beat=True,
        )
        try:
            r = requests.post(url, json=data.dict(), timeout=WORKER_API_TIMEOUT)
        except requests.exceptions.ReadTimeout:
            self.logger.error(
                "Register worker to controller failed for timeout response."
            )
            return

        if r.status_code != 200:
            self.logger.error(
                "Register worker to controller failed for incorrect response."
            )

    def remove_from_controller(self):
        self.logger.info("Remove from controller")

        url = self.controller_addr + "/remove_worker"
        data = RemoveWorkerRequest(worker_id=self.worker_id)
        try:
            r = requests.post(url, json=data.dict(), timeout=WORKER_API_TIMEOUT)
        except requests.exceptions.ReadTimeout:
            self.logger.error(
                "Remove worker from controller failed for timeout response."
            )
            return

        if r.status_code != 200:
            self.logger.error(
                "Remove worker from controller failed for incorrect response."
            )

    def register_heatbeat(self):
        self.add_timer(
            "heartbeat",
            WORKER_HEART_BEAT_INTERVAL,
            self.send_heart_beat,
            args=None,
            kwargs=None,
            workers=1,
        )

    def stop_all_timers(self):
        for name, timer in self.timers.items():
            timer.cancel()
        self.timers.clear()

    def send_heart_beat(self):
        self.logger.info(
            f"Send heart beat. Worker: {self.worker_id}; Address: {self.worker_addr}."
        )

        url = self.controller_addr + "/receive_heart_beat"

        try:
            ret = requests.post(
                url,
                json=WorkerHeartbeatPing(
                    worker_id=self.worker_id,
                ).dict(),
                timeout=WORKER_API_TIMEOUT,
            )
        except requests.exceptions.ReadTimeout:
            self.logger.info(
                f"Failed to send heart beat . Worker: {self.worker_id}; Address: {self.worker_addr}."
            )
            self.register_to_controller()
            return
            
        exist = WorkerHeartbeatPong.parse_obj(ret.json()).exist

        if not exist:
            self.register_to_controller()
