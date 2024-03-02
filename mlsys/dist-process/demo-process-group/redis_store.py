#!/usr/bin/env python3

import atexit
import os
import redis
import subprocess
import time
from datetime import datetime, timedelta
from torch.distributed import Store
from typing import List, Optional, overload


class RedisStoreStats:
    def __init__(self):
        self.req_type = ["get", "set", "add", "compare", "delete", "wait"]
        self.requests = {
            "count": {x: 0 for x in self.req_type},
            "total_delay": {x: 0 for x in self.req_type},
            "max_delay": {x: 0 for x in self.req_type},
            "max_retry": {x: 0 for x in self.req_type},
        }
        self.startup_time = 0

    def reset(self):
        for req in self.req_type:
            self.requests["count"][req] = 0
            self.requests["total_delay"][req] = 0
            self.requests["max_delay"][req] = 0
            self.requests["max_retry"][req] = 0

    def dump(self):
        print(f"Startup time: {self.startup_time}")
        for req in self.req_type:
            if self.requests["count"][req] == 0:
                avg_delay = 0
            else:
                avg_delay = (self.requests["total_delay"][req] /
                             self.requests["count"][req])
            print(f"{req}: count={self.requests['count'][req]} "
                  f"avg_delay={avg_delay*1000:.2f}ms "
                  f"max_delay={self.requests['max_delay'][req]*1000:.2f}ms "
                  f"max_retry={self.requests['max_retry'][req]}")

    def add_delay(self, req: str, delay: float, retry: int=0):
        self.requests["count"][req] += 1
        self.requests["total_delay"][req] += delay
        self.requests["max_delay"][req] = max(self.requests["max_delay"][req], delay)
        self.requests["max_retry"][req] = max(self.requests["max_retry"][req], retry)


class RedisStore(Store):
    def __init__(
        self,
        is_server: bool,
        timeout: timedelta = timedelta(seconds=1200),
        wait_interval: float = 0.1,
        verbose: bool = False,
    ):
        start_time = time.time()
        # call base class constructor
        super().__init__()
        host = os.environ.get("MASTER_ADDR", "")
        port = os.environ.get("MASTER_PORT", "")
        if host == "" or port == "":
            raise ValueError("MASTER_ADDR and MASTER_PORT must be set")
        if is_server:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            redis_path = os.path.join(curr_dir, "../../services/redis")
            redis_bin = os.path.join(redis_path, "redis/src/redis-server")
            redis_conf = os.path.join(redis_path, "redis.conf")
            print(f"Starting Redis server using {redis_bin} {redis_conf} at {host}:{port}")
            proc = subprocess.Popen(
                [redis_bin, redis_conf, "--port", str(port)],
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL,
                shell=False,
            )
            atexit.register(proc.terminate)
        self._timeout = timeout
        self._wait_interval = wait_interval
        self._verbose = verbose
        self._stats = RedisStoreStats()
        time.sleep(self._wait_interval)
        if self._verbose:
            print(f"Connecting to Redis server at {host}:{port}")
        curr_time = time.time()
        self._redis = redis.Redis(host=host, port=port, db=0)
        while True:
            try:
                self._redis.ping()
                break
            except redis.exceptions.ConnectionError:
                if self._verbose:
                    print(f"Failed to ping Redis server at {host}:{port}, retrying...")
                if time.time() - curr_time > self._timeout.total_seconds():
                    raise TimeoutError("Timeout connecting to Redis server")
                time.sleep(0.1)
        if self._verbose:
            print(f"Connected to Redis server at {datetime.now()}")
        self._stats.startup_time = time.time() - start_time

    def reset_stats(self):
        self._stats.reset()

    def dump_stats(self):
        self._stats.dump()

    def dump_keys(self):
        all_keys = self._redis.keys()
        print(f"all {self.num_keys()} keys: {','.join([k.decode('utf-8') for k in all_keys])}")

    def set(self, key: str, value: str):
        start_time = time.time()
        if self._verbose:
            print(f"Setting key {key} to value {value}")
        self._redis.set(key, value)
        self._stats.add_delay("set", time.time() - start_time)

    def get(self, key: str) -> bytes:
        start_time = time.time()
        if self._verbose:
            print(f"Getting key {key}")
        start_time = time.time()
        retry = 0
        value = b""
        while True:
            value = self._redis.get(key)
            if value is not None:
                break
            if time.time() - start_time > self._timeout.total_seconds():
                raise TimeoutError(f"Timeout getting key {key}")
            time.sleep(self._wait_interval)
            retry += 1
        self._stats.add_delay("get", time.time() - start_time, retry)
        return value

    def add(self, key: str, value: int) -> int:
        start_time = time.time()
        res = self._redis.incrby(key, value)
        self._stats.add_delay("add", time.time() - start_time)
        return res

    def compare_set(
        self,
        key: str,
        expected_value: str,
        desired_value: str,
    ) -> bytes:
        start_time = time.time()
        existing_value = self._redis.get(key)
        res = None
        if existing_value is None:
            if expected_value == "":
                self._redis.set(key, desired_value)
                res = desired_value.encode()
            else:
                res = expected_value.encode()
        else:
            if existing_value == expected_value:
                self._redis.set(key, desired_value)
            res = existing_value
        self._stats.add_delay("compare", time.time() - start_time)
        return res

    def delete_key(self, key: str) -> bool:
        start_time = time.time()
        res = self._redis.delete(key) > 0
        self._stats.add_delay("delete", time.time() - start_time)
        return res

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
        start_time = time.time()
        retry = 0
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
