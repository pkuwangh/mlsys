#!/usr/bin/env python3

import redis
import time
from datetime import timedelta
from torch.distributed import Store
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT, _DEFAULT_PG_NCCL_TIMEOUT
from typing import List, overload


class RedisStore(Store):
    def __init__(self, host: str, port: int, is_server: bool, wait_interval: float=0.1):
        # call base class constructor
        super(RedisStore, self).__init__()
        if is_server:
            print("Starting Redis server", flush=True)
        print(f"Connecting to Redis server at {host}:{port}", flush=True)
        self._redis = redis.Redis(host=host, port=port, db=0)
        if isinstance(_DEFAULT_PG_NCCL_TIMEOUT, timedelta):
            self._timeout = _DEFAULT_PG_NCCL_TIMEOUT
        elif isinstance(_DEFAULT_PG_TIMEOUT, timedelta):
            self._timeout = _DEFAULT_PG_TIMEOUT
        else:
            self._timeout = timedelta(seconds=1200)
        self._wait_interval = wait_interval

    def set(self, key: str, value: str):
        self._redis.set(key, value)

    def get(self, key: str) -> bytes:
        retries = [0, 1, self._timeout.total_seconds()]
        for retry_interval in retries:
            if retry_interval > 0:
                time.sleep(retry_interval)
            value = self._redis.get(key)
            if value is not None:
                return value
        raise KeyError(f"Key {key} not found")

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
        return (self._redis.delete(key) > 0)

    def num_keys(self) -> int:
        count = len(self._redis.keys())
        if self._redis.get("key:__rand_int__") is not None:
            count -= 1
        return count

    def set_timeout(self, timeout: timedelta):
        self._timeout = timeout

    @overload
    def wait(self, keys: List[str]):
        self.wait(keys, self._timeout)

    @overload
    def wait(self, keys: List[str], timeout: timedelta):
        start_time = time.time()
        pipe = self._redis.pipeline()
        while len(remaining_keys) > 0:
            for key in remaining_keys:
                pipe.get(key)
            values = pipe.execute()
            remaining_keys = []
            for i, value in enumerate(values):
                if value is None:
                    remaining_keys.append(keys[i])
            curr_time = time.time()
            if (curr_time - start_time) > timeout.total_seconds():
                raise TimeoutError(f"Timeout waiting for keys {remaining_keys}")
            time.sleep(self._wait_interval)


if __name__ == "__main__":
    print(_DEFAULT_PG_NCCL_TIMEOUT)
    print(_DEFAULT_PG_TIMEOUT)
