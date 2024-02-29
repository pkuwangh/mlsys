#!/usr/bin/env python3

import atexit
import os
import redis
import subprocess
import time
from datetime import datetime, timedelta
from torch.distributed import Store
from typing import List, Optional, overload


class RedisStore(Store):
    def __init__(
        self,
        host: str,
        port: int,
        is_server: bool,
        timeout: timedelta = timedelta(seconds=1200),
        wait_interval: float = 0.1,
        verbose: bool = False,
    ):
        # call base class constructor
        super().__init__()
        assert host != "" and port > 0
        if is_server:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            redis_dir = os.path.join(curr_dir, "../../services/redis")
            redis_bin = os.path.join(redis_dir, "redis/src/redis-server")
            redis_conf = os.path.join(redis_dir, "redis.conf")
            print(f"Starting Redis server using {redis_bin} {redis_conf}")
            proc = subprocess.Popen(
                [redis_bin, redis_conf, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            atexit.register(proc.terminate)
            time.sleep(1)
        self._timeout = timeout
        self._wait_interval = wait_interval
        self._verbose = verbose
        if self._verbose:
            print(f"Connecting to Redis server at {host}:{port} at {datetime.now()}")
        self._redis = redis.Redis(host=host, port=port, db=0)
        curr_time = time.time()
        while True:
            try:
                self._redis.ping()
                break
            except redis.exceptions.ConnectionError:
                if self._verbose:
                    print(f"Failed to ping Redis server at {host}:{port}, retrying...")
                if time.time() - curr_time > self._timeout.total_seconds():
                    raise TimeoutError("Timeout connecting to Redis server")
                time.sleep(1)
        if self._verbose:
            print(f"Connected to Redis server at {datetime.now()}")

    def dump_keys(self):
        all_keys = self._redis.keys()
        print(f"all keys: {','.join([k.decode('utf-8') for k in all_keys])}")

    def set(self, key: str, value: str):
        if self._verbose:
            print(f"Setting key {key} to value {value}")
        self._redis.set(key, value)

    def get(self, key: str) -> bytes:
        if self._verbose:
            print(f"Getting key {key}")
        start_time = time.time()
        value = b""
        while True:
            value = self._redis.get(key)
            if value is not None:
                break
            if time.time() - start_time > self._timeout.total_seconds():
                raise TimeoutError(f"Timeout getting key {key}")
            time.sleep(self._wait_interval)
        return value

    def add(self, key: str, value: int) -> int:
        return self._redis.incrby(key, value)

    def compare_set(
        self,
        key: str,
        expected_value: str,
        desired_value: str,
    ) -> bytes:
        existing_value = self._redis.get(key)
        if existing_value is None:
            if expected_value == "":
                self._redis.set(key, desired_value)
                return desired_value.encode()
            else:
                return expected_value.encode()
        else:
            if existing_value == expected_value:
                self._redis.set(key, desired_value)
            return existing_value

    def delete_key(self, key: str) -> bool:
        return self._redis.delete(key) > 0

    def num_keys(self) -> int:
        count = len(self._redis.keys())
        if self._redis.get("key:__rand_int__") is not None:
            count -= 1
        return count

    def set_timeout(self, timeout: timedelta):
        self._timeout = timeout

    @overload
    def wait(self, keys: List[str]):
        ...

    @overload
    def wait(self, keys: List[str], timeout: timedelta):
        ...

    def wait(self, keys: List[str], timeout: Optional[timedelta]=None):
        if self._verbose:
            print(f"Waiting for keys {keys}")
        eff_timeout = self._timeout if timeout is None else timeout
        start_time = time.time()
        pipe = self._redis.pipeline()
        remaining_keys = [x for x in keys]
        while len(remaining_keys) > 0:
            for key in remaining_keys:
                pipe.get(key)
            values = pipe.execute()
            remaining_keys = []
            for i, value in enumerate(values):
                if value is None:
                    remaining_keys.append(keys[i])
            curr_time = time.time()
            if (curr_time - start_time) > eff_timeout.total_seconds():
                raise TimeoutError(f"Timeout waiting for keys {remaining_keys}")
            time.sleep(self._wait_interval)
