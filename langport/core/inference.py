"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Any, Dict, Iterable, List, Optional
import sys
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from langport.constants import ErrorCode
from langport.core.generation_worker import (
    GenerationTask,
    GenerationTaskStreamOutput,
    UsageInfo,
)

from langport.data.conversation import get_conv_template, SeparatorStyle
from langport.model.model_adapter import load_model, get_conversation_template


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def inference_batch_step(model, input_ids, encoder_output, past_key_values):
    if model.config.is_encoder_decoder:
        out = model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_output,
            use_cache=True,
            past_key_values=past_key_values,
        )

        logits = model.lm_head(out[0])
    else:
        out = model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits = out.logits
    past_key_values = out.past_key_values
    return logits, past_key_values


@torch.inference_mode()
def generate_batch_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: List[GenerationTask],
    device: str,
    context_len: int = 2048,
    stream_interval: int = 2,
):
    prompts = [task.prompt for task in tasks]
    encoded_tokens = tokenizer(
        prompts, padding="longest", return_tensors="pt", return_length=True
    ).input_ids
    input_ids = encoded_tokens["input_ids"]
    length = encoded_tokens["length"]

    # check length
    for task_i, task in enumerate(tasks):
        token_num = length[task_i]
        max_new_tokens = task.max_new_tokens
        if token_num + max_new_tokens > context_len:
            yield {
                "task_id": task.max_new_tokens,
                "message": f"This model's maximum context length is {context_len} tokens. "
                f"However, you requested {max_new_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{max_new_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.",
                "error_code": ErrorCode.CONTEXT_OVERFLOW,
            }
    # logits_processors
    logits_processors = []
    for task_i, task in enumerate(tasks):
        temperature = task.temperature
        repetition_penalty = task.repetition_penalty
        top_p = task.top_p
        top_k = task.top_k

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        logits_processors.append(logits_processor)

    # prepare context
    input_echo_len = len(input_ids)
    output_ids = input_ids

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        input_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        encoder_output = None
        input_ids = input_ids

    # infer
    past_key_values = None
    for i in range(max_new_tokens):
        logits, past_key_values = inference_batch_step(
            model=model,
            input_ids=input_ids,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
        )

        last_tokens = []
        for task_i, task in enumerate(tasks):
            sub_logits = logits[task_i, -1, :]
            len_prompt = len(task.prompt)
            stop_str = task.stop
            echo = task.echo
            stop_token_ids = task.stop_token_ids
            stop_token_ids.append(tokenizer.eos_token_id)
            logits_processor = logits_processors[task_i]

            if repetition_penalty > 1.0:
                tmp_output_ids = output_ids
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, sub_logits)[0]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits, dim=1))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            last_tokens.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                    else:
                        raise ValueError("Invalid stop field type.")

                yield GenerationTaskStreamOutput(
                    task_id=task.task_id,
                    text=output,
                    finish_reason=None,
                    usage=UsageInfo(
                        prompt_tokens=input_echo_len,
                        completion_tokens=i,
                        total_tokens=input_echo_len + i,
                    ),
                    error_code=0,
                    message="",
                )

            if stopped:
                break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
