import os
import sys
import math
import multiprocessing
from abc import ABC
from typing import (
    List,
    Optional,
    Generator,
    Sequence,
    Deque,
    Tuple,
    Callable,
)
from collections import deque, OrderedDict

from . import llama_cpp

import numpy as np
import numpy.typing as npt


class LlamaCache(ABC):
    """Base cache class for a llama.cpp model."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        pass

    @property
    def cache_size(self):
        return 0

    def _find_longest_prefix_key(
            self,
            key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        pass

    def __getitem__(self, key: Sequence[int]) -> "LlamaState":
        pass

    def __contains__(self, key: Sequence[int]) -> bool:
        pass

    def __setitem__(self, key: Sequence[int], value: "LlamaState"):
        pass


class LlamaRAMCache(LlamaCache):
    """Cache for a llama.cpp model using RAM."""

    def __init__(self, capacity_bytes: int = (2 << 30)):
        super().__init__(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.cache_state: OrderedDict[Tuple[int, ...], "LlamaState"] = OrderedDict()

    @property
    def cache_size(self):
        return sum([state.llama_state_size for state in self.cache_state.values()])

    def _find_longest_prefix_key(
            self,
            key: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        min_len = 0
        min_key = None
        keys = (
            (k, Llama.longest_token_prefix(k, key)) for k in self.cache_state.keys()
        )
        for k, prefix_len in keys:
            if prefix_len > min_len:
                min_len = prefix_len
                min_key = k
        return min_key

    def __getitem__(self, key: Sequence[int]) -> "LlamaState":
        key = tuple(key)
        _key = self._find_longest_prefix_key(key)
        if _key is None:
            raise KeyError("Key not found")
        value = self.cache_state[_key]
        self.cache_state.move_to_end(_key)
        return value

    def __contains__(self, key: Sequence[int]) -> bool:
        return self._find_longest_prefix_key(tuple(key)) is not None

    def __setitem__(self, key: Sequence[int], value: "LlamaState"):
        key = tuple(key)
        if key in self.cache_state:
            del self.cache_state[key]
        self.cache_state[key] = value
        while self.cache_size > self.capacity_bytes:
            self.cache_state.popitem(last=False)


class LlamaState:
    def __init__(
            self,
            eval_tokens: Deque[int],
            eval_logits: Deque[List[float]],
            input_ids: npt.NDArray[np.intc],
            scores: npt.NDArray[np.single],
            llama_state,  # type: llama_cpp.Array[llama_cpp.c_uint8]
            llama_state_size: int,
    ):
        self.eval_tokens = eval_tokens
        self.eval_logits = eval_logits
        self.input_ids = input_ids
        self.scores = scores
        self.llama_state = llama_state
        self.llama_state_size = llama_state_size


LogitsProcessor = Callable[[List[int], List[float]], List[float]]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


StoppingCriteria = Callable[[List[int], List[float]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(self, input_ids: List[int], logits: List[float]) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    def __init__(
            self,
            model_path: str,
            # NOTE: These parameters are likely to change in the future.
            n_ctx: int = 512,
            n_parts: int = -1,
            n_gpu_layers: int = 0,
            seed: int = 1337,
            f16_kv: bool = True,
            logits_all: bool = False,
            vocab_only: bool = False,
            use_mmap: bool = True,
            use_mlock: bool = False,
            embedding: bool = False,
            n_threads: Optional[int] = None,
            n_batch: int = 512,
            last_n_tokens_size: int = 64,
            lora_base: Optional[str] = None,
            lora_path: Optional[str] = None,
            verbose: bool = False,
    ):
        """Load a llama.cpp model from `model_path`.

        Args:
            model_path: Path to the model.
            n_ctx: Maximum context size.
            n_parts: Number of parts to split the model into. If -1, the number of parts is automatically determined.
            seed: Random seed. 0 for random.
            f16_kv: Use half-precision for key/value cache.
            logits_all: Return logits for all tokens, not just the last token.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.verbose = verbose
        self.model_path = model_path

        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = n_ctx
        self.params.n_gpu_layers = n_gpu_layers
        self.params.seed = seed
        self.params.f16_kv = f16_kv
        self.params.logits_all = logits_all
        self.params.vocab_only = vocab_only
        self.params.use_mmap = use_mmap if lora_path is None else False
        self.params.use_mlock = use_mlock
        self.params.embedding = embedding

        self.last_n_tokens_size = last_n_tokens_size
        self.n_batch = min(n_ctx, n_batch)
        self.eval_tokens: Deque[int] = deque(maxlen=n_ctx)
        self.eval_logits: Deque[List[float]] = deque(maxlen=n_ctx if logits_all else 1)

        self.cache: Optional[LlamaCache] = None

        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)

        self.lora_base = lora_base
        self.lora_path = lora_path

        ### DEPRECATED ###
        self.n_parts = n_parts
        ### DEPRECATED ###

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self.ctx = llama_cpp.llama_init_from_file(
            self.model_path.encode("utf-8"), self.params
        )

        assert self.ctx is not None

        if self.lora_path:
            if llama_cpp.llama_apply_lora_from_file(
                    self.ctx,
                    llama_cpp.c_char_p(self.lora_path.encode("utf-8")),
                    llama_cpp.c_char_p(self.lora_base.encode("utf-8"))
                    if self.lora_base is not None
                    else llama_cpp.c_char_p(0),
                    llama_cpp.c_int(self.n_threads),
            ):
                raise RuntimeError(
                    f"Failed to apply LoRA from lora path: {self.lora_path} to base path: {self.lora_base}"
                )

        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()
        size = llama_cpp.c_size_t(self._n_vocab)
        sorted = llama_cpp.c_bool(False)
        self._candidates_data = np.array(
            [],
            dtype=np.dtype(
                [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
            ),
        )
        self._candidates_data.resize(3, self._n_vocab, refcheck=False)
        candidates = llama_cpp.llama_token_data_array(
            data=self._candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
            size=size,
            sorted=sorted,
        )
        self._candidates = candidates
        self._token_nl = Llama.token_nl()
        self._token_eos = Llama.token_eos()

        self._input_ids = np.array([], dtype=np.intc)
        self._scores = np.ndarray((0, self._n_vocab), dtype=np.single)

    def tokenize(self, text: bytes, add_bos: bool = True) -> List[int]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        assert self.ctx is not None
        n_ctx = self._n_ctx
        tokens = (llama_cpp.llama_token * n_ctx)()
        n_tokens = llama_cpp.llama_tokenize(
            self.ctx,
            text,
            tokens,
            llama_cpp.c_int(n_ctx),
            llama_cpp.c_bool(add_bos),
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.ctx,
                text,
                tokens,
                llama_cpp.c_int(n_tokens),
                llama_cpp.c_bool(add_bos),
            )
            if n_tokens < 0:
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )
        return list(tokens[:n_tokens])

    def detokenize(self, tokens: List[int]) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        assert self.ctx is not None
        output = b""
        for token in tokens:
            output += llama_cpp.llama_token_to_str(
                self.ctx, llama_cpp.llama_token(token)
            )
        return output

    def set_cache(self, cache: Optional[LlamaCache]):
        """Set the cache.

        Args:
            cache: The cache to set.
        """
        self.cache = cache

    def reset(self):
        """Reset the model state."""
        self.eval_tokens.clear()
        self.eval_logits.clear()
        self._input_ids = np.array([], dtype=np.intc)
        self._scores = np.ndarray((0, self._n_vocab), dtype=np.single)

    def eval(self, tokens: Sequence[int]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        assert self.ctx is not None
        n_ctx = self._n_ctx
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i: min(len(tokens), i + self.n_batch)]
            n_past = min(n_ctx - len(batch), len(self._input_ids))
            n_tokens = len(batch)
            return_code = llama_cpp.llama_eval(
                ctx=self.ctx,
                tokens=(llama_cpp.llama_token * len(batch))(*batch),
                n_tokens=llama_cpp.c_int(n_tokens),
                n_past=llama_cpp.c_int(n_past),
                n_threads=llama_cpp.c_int(self.n_threads),
            )
            if return_code != 0:
                raise RuntimeError(f"llama_eval returned {return_code}")
            # Save tokens
            self.eval_tokens.extend(batch)
            self._input_ids: npt.NDArray[np.intc] = np.concatenate(
                (self._input_ids, np.array(batch, dtype=np.intc)), axis=0
            )
            # Save logits
            rows = n_tokens if self.params.logits_all else 1
            n_vocab = self._n_vocab
            cols = n_vocab
            logits_view = llama_cpp.llama_get_logits(self.ctx)
            logits = [logits_view[i * cols: (i + 1) * cols] for i in range(rows)]
            self.eval_logits.extend(logits)
            self._scores: npt.NDArray[np.single] = np.concatenate(
                (self._scores, np.array(logits, dtype=np.single)), axis=0
            )

    def _sample(
            self,
            last_n_tokens_data,  # type: llama_cpp.Array[llama_cpp.llama_token]
            last_n_tokens_size: llama_cpp.c_int,
            top_k: llama_cpp.c_int,
            top_p: llama_cpp.c_float,
            temp: llama_cpp.c_float,
            tfs_z: llama_cpp.c_float,
            repeat_penalty: llama_cpp.c_float,
            frequency_penalty: llama_cpp.c_float,
            presence_penalty: llama_cpp.c_float,
            mirostat_mode: llama_cpp.c_int,
            mirostat_tau: llama_cpp.c_float,
            mirostat_eta: llama_cpp.c_float,
            penalize_nl: bool = True,
            logits_processor: Optional[LogitsProcessorList] = None,
    ):
        assert self.ctx is not None
        assert len(self.eval_logits) > 0
        assert self._scores.shape[0] > 0
        n_vocab = self._n_vocab
        n_ctx = self._n_ctx
        top_k = llama_cpp.c_int(n_vocab) if top_k.value <= 0 else top_k
        last_n_tokens_size = (
            llama_cpp.c_int(n_ctx)
            if last_n_tokens_size.value < 0
            else last_n_tokens_size
        )
        logits: npt.NDArray[np.single] = self._scores[-1, :]

        if logits_processor is not None:
            logits = np.array(
                logits_processor(self._input_ids.tolist(), logits.tolist()),
                dtype=np.single,
            )
            self._scores[-1, :] = logits
            self.eval_logits[-1] = logits.tolist()

        nl_logit = logits[self._token_nl]
        candidates = self._candidates
        candidates_data = self._candidates_data
        candidates_data["id"] = np.arange(n_vocab, dtype=np.intc)  # type: ignore
        candidates_data["logit"] = logits
        candidates_data["p"] = np.zeros(n_vocab, dtype=np.single)
        candidates.data = candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p)
        candidates.sorted = llama_cpp.c_bool(False)
        candidates.size = llama_cpp.c_size_t(n_vocab)
        llama_cpp.llama_sample_repetition_penalty(
            ctx=self.ctx,
            last_tokens_data=last_n_tokens_data,
            last_tokens_size=last_n_tokens_size,
            candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            penalty=repeat_penalty,
        )
        llama_cpp.llama_sample_frequency_and_presence_penalties(
            ctx=self.ctx,
            candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            last_tokens_data=last_n_tokens_data,
            last_tokens_size=last_n_tokens_size,
            alpha_frequency=frequency_penalty,
            alpha_presence=presence_penalty,
        )
        if not penalize_nl:
            candidates.data[self._token_nl].logit = llama_cpp.c_float(nl_logit)
        if temp.value == 0.0:
            return llama_cpp.llama_sample_token_greedy(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            )
        elif mirostat_mode.value == 1:
            mirostat_mu = llama_cpp.c_float(2.0 * mirostat_tau.value)
            mirostat_m = llama_cpp.c_int(100)
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                temp=temp,
            )
            return llama_cpp.llama_sample_token_mirostat(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=llama_cpp.ctypes.byref(mirostat_mu),  # type: ignore
                m=mirostat_m,
            )
        elif mirostat_mode.value == 2:
            mirostat_mu = llama_cpp.c_float(2.0 * mirostat_tau.value)
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.pointer(candidates),
                temp=temp,
            )
            return llama_cpp.llama_sample_token_mirostat_v2(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=llama_cpp.ctypes.byref(mirostat_mu),  # type: ignore
            )
        else:
            llama_cpp.llama_sample_top_k(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                k=top_k,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_tail_free(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                z=tfs_z,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_typical(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                p=llama_cpp.c_float(1.0),
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_top_p(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                p=top_p,
                min_keep=llama_cpp.c_size_t(1),
            )
            llama_cpp.llama_sample_temperature(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
                temp=temp,
            )
            return llama_cpp.llama_sample_token(
                ctx=self.ctx,
                candidates=llama_cpp.ctypes.byref(candidates),  # type: ignore
            )

    def sample(
            self,
            top_k: int = 40,
            top_p: float = 0.95,
            temp: float = 0.80,
            repeat_penalty: float = 1.1,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_eta: float = 0.1,
            mirostat_tau: float = 5.0,
            penalize_nl: bool = True,
            logits_processor: Optional[LogitsProcessorList] = None,
    ):
        """Sample a token from the model.

        Args:
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.

        Returns:
            The sampled token.
        """
        assert self.ctx is not None
        last_n_tokens_data = [llama_cpp.llama_token(0)] * max(
            0, self.last_n_tokens_size - len(self._input_ids)
        ) + self._input_ids[-self.last_n_tokens_size:].tolist()
        return self._sample(
            last_n_tokens_data=(llama_cpp.llama_token * self.last_n_tokens_size)(
                *last_n_tokens_data
            ),
            last_n_tokens_size=llama_cpp.c_int(self.last_n_tokens_size),
            top_k=llama_cpp.c_int(top_k),
            top_p=llama_cpp.c_float(top_p),
            temp=llama_cpp.c_float(temp),
            tfs_z=llama_cpp.c_float(tfs_z),
            repeat_penalty=llama_cpp.c_float(repeat_penalty),
            frequency_penalty=llama_cpp.c_float(frequency_penalty),
            presence_penalty=llama_cpp.c_float(presence_penalty),
            mirostat_mode=llama_cpp.c_int(mirostat_mode),
            mirostat_tau=llama_cpp.c_float(mirostat_tau),
            mirostat_eta=llama_cpp.c_float(mirostat_eta),
            penalize_nl=penalize_nl,
            logits_processor=logits_processor,
        )

    def generate(
            self,
            tokens: Sequence[int],
            top_k: int = 40,
            top_p: float = 0.95,
            temp: float = 0.80,
            repeat_penalty: float = 1.1,
            reset: bool = True,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        assert self.ctx is not None

        if reset and len(self._input_ids) > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                if self.verbose:
                    print("Llama.generate: prefix-match hit", file=sys.stderr)
                reset = False
                tokens = tokens[longest_prefix:]
                self._input_ids = self._input_ids[:longest_prefix]
                self._scores = self._scores[:longest_prefix, :]
                for _ in range(len(self.eval_tokens) - longest_prefix):
                    self.eval_tokens.pop()
                    try:
                        self.eval_logits.pop()
                    except IndexError:
                        pass

        if reset:
            self.reset()

        while True:
            self.eval(tokens)
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                logits_processor=logits_processor,
            )
            if stopping_criteria is not None and stopping_criteria(
                    self._input_ids.tolist(), self._scores[-1, :].tolist()
            ):
                return
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

    def embed(
            self, input: str, model: Optional[str] = None
    ) -> List[float]:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            A list of embeddings
        """
        assert self.ctx is not None
        model_name: str = model if model is not None else self.model_path

        if self.params.embedding == False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        tokens = self.tokenize(input.encode("utf-8"))
        self.reset()
        self.eval(tokens)
        n_tokens = len(tokens)
        total_tokens += n_tokens
        embedding = llama_cpp.llama_get_embeddings(self.ctx)[
                    : llama_cpp.llama_n_embd(self.ctx)
                    ]

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        return embedding
    

    def __del__(self):
        if self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None

    def __getstate__(self):
        return dict(
            verbose=self.verbose,
            model_path=self.model_path,
            n_ctx=self.params.n_ctx,
            n_gpu_layers=self.params.n_gpu_layers,
            seed=self.params.seed,
            f16_kv=self.params.f16_kv,
            logits_all=self.params.logits_all,
            vocab_only=self.params.vocab_only,
            use_mmap=self.params.use_mmap,
            use_mlock=self.params.use_mlock,
            embedding=self.params.embedding,
            last_n_tokens_size=self.last_n_tokens_size,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            lora_base=self.lora_base,
            lora_path=self.lora_path,
            ### DEPRECATED ###
            n_parts=self.n_parts,
            ### DEPRECATED ###
        )

    def __setstate__(self, state):
        self.__init__(
            model_path=state["model_path"],
            n_ctx=state["n_ctx"],
            n_parts=state["n_parts"],
            n_gpu_layers=state["n_gpu_layers"],
            seed=state["seed"],
            f16_kv=state["f16_kv"],
            logits_all=state["logits_all"],
            vocab_only=state["vocab_only"],
            use_mmap=state["use_mmap"],
            use_mlock=state["use_mlock"],
            embedding=state["embedding"],
            n_threads=state["n_threads"],
            n_batch=state["n_batch"],
            last_n_tokens_size=state["last_n_tokens_size"],
            lora_base=state["lora_base"],
            lora_path=state["lora_path"],
            verbose=state["verbose"],
        )

    def save_state(self) -> LlamaState:
        assert self.ctx is not None
        state_size = llama_cpp.llama_get_state_size(self.ctx)
        llama_state = (llama_cpp.c_uint8 * int(state_size))()
        n_bytes = llama_cpp.llama_copy_state_data(self.ctx, llama_state)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        llama_state_compact = (llama_cpp.c_uint8 * int(n_bytes))()
        llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))
        if self.verbose:
            print(
                f"Llama.save_state: saving {n_bytes} bytes of llama state",
                file=sys.stderr,
            )
        return LlamaState(
            eval_tokens=self.eval_tokens.copy(),
            eval_logits=self.eval_logits.copy(),
            scores=self._scores.copy(),
            input_ids=self._input_ids.copy(),
            llama_state=llama_state_compact,
            llama_state_size=n_bytes,
        )

    def load_state(self, state: LlamaState) -> None:
        assert self.ctx is not None
        self.eval_tokens = state.eval_tokens.copy()
        self.eval_logits = state.eval_logits.copy()
        self._scores = state.scores.copy()
        self._input_ids = state.input_ids.copy()
        state_size = state.llama_state_size
        if llama_cpp.llama_set_state_data(self.ctx, state.llama_state) != state_size:
            raise RuntimeError("Failed to set llama state data")

    def n_ctx(self) -> int:
        """Return the context window size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_ctx(self.ctx)

    def n_embd(self) -> int:
        """Return the embedding size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_embd(self.ctx)

    def n_vocab(self) -> int:
        """Return the vocabulary size."""
        assert self.ctx is not None
        return llama_cpp.llama_n_vocab(self.ctx)

    def tokenizer(self) -> "LlamaTokenizer":
        """Return the tokenizer for this model."""
        assert self.ctx is not None
        return LlamaTokenizer(self)

    @staticmethod
    def token_eos() -> int:
        """Return the end-of-sequence token."""
        return llama_cpp.llama_token_eos()

    @staticmethod
    def token_bos() -> int:
        """Return the beginning-of-sequence token."""
        return llama_cpp.llama_token_bos()

    @staticmethod
    def token_nl() -> int:
        """Return the newline token."""
        return llama_cpp.llama_token_nl()

    @staticmethod
    def logits_to_logprobs(logits: List[float]) -> List[float]:
        exps = [math.exp(float(x)) for x in logits]
        sum_exps = sum(exps)
        return [math.log(x / sum_exps) for x in exps]

    @staticmethod
    def longest_token_prefix(a: Sequence[int], b: Sequence[int]):
        longest_prefix = 0
        for _a, _b in zip(a, b):
            if _a == _b:
                longest_prefix += 1
            else:
                break
        return longest_prefix


class LlamaTokenizer:
    def __init__(self, llama: Llama):
        self.llama = llama

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        return self.llama.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos
        )

    def decode(self, tokens: List[int]) -> str:
        return self.llama.detokenize(tokens).decode("utf-8", errors="ignore")

    @classmethod
    def from_ggml_file(cls, path: str) -> "LlamaTokenizer":
        return cls(Llama(model_path=path, vocab_only=True))