"""Executable caching and compilation accounting for the Wan CuTe kernels.

Counters describe this process: a cache hit reuses an executable without calling
cute.compile. Compile time measures cute.compile itself, including any DSL disk
cache lookup/loading; it excludes tensor allocation and kernel execution.
"""

import os
from dataclasses import dataclass
from functools import lru_cache, wraps
from time import perf_counter

_caches = []
_compiled = 0
_compile_seconds = 0.0


def compile_cache(*, maxsize=None):
    """Register an LRU cache; callers must use consistent argument forms/order.

    No signature binding is added to the launch path. Precompile helpers use
    the same explicit arguments as runtime calls.
    """

    def decorate(function):
        @wraps(function)
        def compile_variant(*args, **kwargs):
            if os.environ.get("WAN_VAE_COMPILE_KEYS") == "1":
                print(f"CuTe cache miss: {function.__module__}.{function.__name__}{args!r} {kwargs!r}", flush=True)
            before = compile_stats()
            executable = function(*args, **kwargs)
            delta = compile_stats() - before
            cached.compile_count += delta.compiled
            cached.compile_seconds += delta.seconds
            return executable

        cached = lru_cache(maxsize=maxsize)(compile_variant)
        cached.compile_count = 0
        cached.compile_seconds = 0.0
        _caches.append(cached)
        return cached

    return decorate


def compile_kernel(*args, **kwargs):
    """Time a successful CuTe compilation without intercepting global DSL APIs."""
    from cutlass import cute

    global _compiled, _compile_seconds
    start = perf_counter()
    executable = cute.compile(*args, **kwargs)
    _compile_seconds += perf_counter() - start
    _compiled += 1
    return executable


@dataclass(frozen=True)
class CompileStats:
    compiled: int
    cache_hits: int
    seconds: float

    def __sub__(self, other):
        return CompileStats(
            self.compiled - other.compiled,
            self.cache_hits - other.cache_hits,
            self.seconds - other.seconds,
        )


def compile_stats() -> CompileStats:
    """Snapshot counters; do not clear executable caches inside a measurement."""
    return CompileStats(_compiled, sum(cache.cache_info().hits for cache in _caches), _compile_seconds)


def compile_family_stats() -> dict[str, CompileStats]:
    """Attribute successful compile calls and cache hits to their kernel factory."""
    return {
        f"{cache.__module__.rsplit('.', 1)[-1]}.{cache.__name__}": CompileStats(
            cache.compile_count, cache.cache_info().hits, cache.compile_seconds
        )
        for cache in _caches
    }
