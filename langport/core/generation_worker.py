import argparse
import asyncio
import dataclasses
import logging
import json
import os
import time
from typing import List, Optional, Union
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from tenacity import retry, stop_after_attempt

from langport.protocol.worker_protocol import (
    RegisterWorkerRequest,
    RemoveWorkerRequest,
    WorkerHeartbeat,
    WorkerStatus,
)

import torch

from langport.constants import (
    WORKER_API_TIMEOUT,
    WORKER_HEART_BEAT_INTERVAL,
    WORKER_HEART_BEAT_CHECK_INTERVAL,
    WORKER_INFERENCE_TIMER_INTERVAL,
    ErrorCode,
)
from langport.model.model_adapter import load_model
from langport.core.inference import generate_stream, generate_batch_stream
from langport.utils import server_error_msg, pretty_print_semaphore
