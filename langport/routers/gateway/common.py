
import json
from typing import Optional, Union
import httpx
import numpy as np

BASE_SETTINGS = False
if not BASE_SETTINGS:
    try:
        from pydantic import BaseSettings
        BASE_SETTINGS = True
    except ImportError:
        BASE_SETTINGS = False

if not BASE_SETTINGS:
    try:
        from pydantic_settings import BaseSettings
        BASE_SETTINGS = True
    except ImportError:
        BASE_SETTINGS = False

if not BASE_SETTINGS:
    raise Exception("Cannot import BaseSettings from pydantic or pydantic-settings")

from langport.core.dispatch import DispatchMethod
from langport.protocol.openai_api_protocol import ErrorResponse
from langport.protocol.worker_protocol import WorkerAddressRequest, WorkerAddressResponse

from typing import Generator, Optional, Union, Dict, List, Any

from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction

from langport.constants import WORKER_API_TIMEOUT, ErrorCode
from langport.model.model_adapter import get_conversation_template
from fastapi.exceptions import RequestValidationError

class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"

LANGPORT_HEADER = {"User-Agent": "Langport API Server"}

def create_server_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )

def create_bad_request_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )


async def _get_worker_address(
    app_settings: AppSettings, model_name: str, feature: str, client: httpx.AsyncClient, dispatch: Union[str, DispatchMethod]
) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param feature: The worker's feature
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    if isinstance(dispatch, str):
        dispatch = DispatchMethod.from_str(dispatch)
    if dispatch == DispatchMethod.LOTTERY:
        payload = WorkerAddressRequest(
            condition=f"{{model_name}}=='{model_name}' and '{feature}' in {{features}}", expression="1 / 0.01 + {speed}"
        )
    elif dispatch == DispatchMethod.SHORTEST_QUEUE:
        payload = WorkerAddressRequest(
            condition=f"{{model_name}}=='{model_name}' and '{feature}' in {{features}}", expression="{queue_length}/{speed}"
        )
    else:
        raise Exception("Error dispatch method.")
    ret = await client.post(
        controller_address + "/get_worker_address",
        json=payload.dict(),
    )
    response = WorkerAddressResponse.parse_obj(ret.json())
    address_list = response.address_list
    values = [json.loads(obj) for obj in response.values]

    # sort
    sorted_result = sorted(zip(address_list, values), key=lambda x: x[1])
    address_list = [x[0] for x in sorted_result]
    values = [x[1] for x in sorted_result]

    # No available worker
    if address_list == []:
        raise ValueError(f"No available worker for {model_name} and {feature}")
    if dispatch == DispatchMethod.LOTTERY:
        node_speeds = np.array(values, dtype=np.float32)
        norm = np.sum(node_speeds)
        if norm < 1e-4:
            return ""
        node_speeds = node_speeds / norm
        pt = np.random.choice(np.arange(len(address_list)), p=node_speeds)
        worker_addr = address_list[pt]
    elif dispatch == DispatchMethod.SHORTEST_QUEUE:
        worker_addr = address_list[0]
    else:
        raise Exception("Error dispatch method.")
    # logger.debug(f"model_name: {model_name}, feature: {feature}, worker_addr: {worker_addr}")
    return worker_addr


async def _list_models(app_settings: AppSettings, feature: Optional[str], client: httpx.AsyncClient) -> str:
    controller_address = app_settings.controller_address

    if feature is None:
        condition = "True"
    else:
        condition=f"'{feature}' in {{features}}"
    payload = WorkerAddressRequest(
        condition=condition, expression="{model_name}"
    )
    try:
        ret = await client.post(
            controller_address + "/get_worker_address",
            json=payload.dict(),
        )
        if ret.status_code != 200:
            return []
    except Exception as e:
        print("[Exception] list model: ", e)
        return []
    response = WorkerAddressResponse.parse_obj(ret.json())
    
    address_list = response.address_list
    models = [json.loads(obj) for obj in response.values]
    # No available worker
    if address_list == []:
        # raise ValueError(f"No available worker for feature {feature}")
        return []

    return models

async def check_model(app_settings: AppSettings, request, feature: str, model_name: str) -> Optional[JSONResponse]:
    ret = None
    async with httpx.AsyncClient() as client:
        try:
            models = await _list_models(app_settings, feature, client)
        except Exception as e:
            ret = create_bad_request_response(
                ErrorCode.INVALID_MODEL,
                str(e),
            )
            return ret
        if len(models) == 0 or model_name not in models:
            models_unique = list(set(models))
            ret = create_bad_request_response(
                ErrorCode.INVALID_MODEL,
                f"Only {'&&'.join(models_unique)} allowed now, your model {model_name}",
            )
    return ret

def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )
    if request.presence_penalty is not None and request.presence_penalty < -2.0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.presence_penalty} is less than the minimum of -2.0 - 'presence_penalty'",
        )
    if request.presence_penalty is not None and request.presence_penalty > 2.0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.presence_penalty} is less than the maximum of 2.0 - 'presence_penalty'",
        )
    if request.frequency_penalty is not None and request.frequency_penalty < -2.0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.frequency_penalty} is less than the minimum of -2.0 - 'frequency_penalty'",
        )
    if request.frequency_penalty is not None and request.frequency_penalty > 2.0:
        return create_bad_request_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.frequency_penalty} is less than the maximum of 2.0 - 'frequency_penalty'",
        )

    return None

