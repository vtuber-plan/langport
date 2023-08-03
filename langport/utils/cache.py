from functools import lru_cache as python_lru_cache
import hashlib
import pickle

def hash_list(l: list) -> int:
    __hash = 0
    for i, e in enumerate(l):
        __hash = hash((__hash, i, hash_item(e)))
    return __hash

def hash_dict(d: dict) -> int:
    __hash = 0
    for k, v in d.items():
        __hash = hash((__hash, k, hash_item(v)))
    return __hash

def hash_item(e) -> int:
    if hasattr(e, '__hash__') and callable(e.__hash__):
        try:
            return hash(e)
        except TypeError:
            pass
    if isinstance(e, (list, set, tuple)):
        return hash_list(list(e))
    elif isinstance(e, (dict)):
        return hash_dict(e)
    else:
        raise TypeError(f'unhashable type: {e.__class__}')

def lru_cache(*opts, **kwopts):
    def decorator(func):
        def wrapper(*args, **kwargs):
            __hash = hash_item([id(func)] + list(args) + list(kwargs.items()))
            print([id(func)] + list(args) + list(kwargs.items()))

            @python_lru_cache(*opts, **kwopts)
            def cached_func(args_hash):
                return func(*args, **kwargs)
            
            return cached_func(__hash)
        return wrapper
    return decorator