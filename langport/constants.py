from enum import IntEnum


CONTROLLER_HEART_BEAT_EXPIRATION = 90
CONTROLLER_STATUS_FRESH_INTERVAL = 30
WORKER_HEART_BEAT_INTERVAL = 30
EMBEDDING_INFERENCE_INTERVAL = 0.5
GENERATION_INFERENCE_INTERVAL = 0.5
WORKER_API_TIMEOUT = 20

LOGDIR = "./logs"

class ErrorCode(IntEnum):
    OK = 200
    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
