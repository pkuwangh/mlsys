#!/usr/bin/env python3

import atexit
import hashlib
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
            "total_interval": {x: 0 for x in self.req_type},
            "max_interval": {x: 0 for x in self.req_type},
            "max_interval_idx": {x: 0 for x in self.req_type},
        }
        self.startup_time = 0
        self.last_request_ts = time.time()

    def reset(self):
        self.startup_time = 0
        for req in self.req_type:
            self.requests["count"][req] = 0
            self.requests["total_delay"][req] = 0
            self.requests["max_delay"][req] = 0
            self.requests["max_retry"][req] = 0
            self.requests["total_interval"][req] = 0
            self.requests["max_interval"][req] = 0
            self.requests["max_interval_idx"][req] = 0

    def dump(self):
        no_stats = True
        if self.startup_time > 0:
            print(f"redis-client startup time: {self.startup_time}")
            no_stats = False
        for req in self.req_type:
            count = self.requests["count"][req]
            if count == 0:
                continue
            avg_delay = self.requests["total_delay"][req] / count
            avg_interval = self.requests["total_interval"][req] / count
            print(
                f"redis-client {req}: count={self.requests['count'][req]} "
                f"avg_delay={avg_delay*1000:.2f}ms "
                f"max_delay={self.requests['max_delay'][req]*1000:.2f}ms "
                f"max_retry={self.requests['max_retry'][req]} "
                f"avg_interval={avg_interval*1000:.2f}ms "
                f"max_interval={self.requests['max_interval'][req]*1000:.2f}ms "
                f"max_interval_idx={self.requests['max_interval_idx'][req]}"
            )
            no_stats = False
        if no_stats:
            print("redis-client no activity")

    def add_delay(self, req: str, delay: float, retry: int = 0):
        self.requests["count"][req] += 1
        self.requests["total_delay"][req] += delay
        self.requests["max_delay"][req] = max(self.requests["max_delay"][req], delay)
        self.requests["max_retry"][req] = max(self.requests["max_retry"][req], retry)
        curr_ts = time.time()
        interval = curr_ts - self.last_request_ts - delay
        self.requests["total_interval"][req] += interval
        if interval > self.requests["max_interval"][req]:
            self.requests["max_interval"][req] = interval
            self.requests["max_interval_idx"][req] = self.requests["count"][req]
        self.last_request_ts = curr_ts


class RedisStore(Store):
    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

    def __init__(
        self,
        is_server: bool,
        timeout: timedelta = timedelta(seconds=1200),
        num_shards: int = 1,
        max_connections: int = 1,
        wait_interval: float = 0.1,
        verbose: int = 0,
    ):
        start_time = time.time()
        # call base class constructor
        super().__init__()
        host = os.environ.get("MASTER_ADDR", "")
        port = os.environ.get("MASTER_PORT", "")
        if host == "" or port == "":
            raise ValueError("MASTER_ADDR and MASTER_PORT must be set")
        self._ports = [int(port) + i for i in range(num_shards)]
        if is_server:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            redis_path = os.path.join(curr_dir, "../../services/redis")
            redis_bin = os.path.join(redis_path, "redis/src/redis-server")
            redis_conf = os.path.join(redis_path, "redis.conf")
            print(
                f"Starting {num_shards} Redis server using "
                f"{redis_bin} {redis_conf} on {host}"
            )
            procs = []
            for i in range(num_shards):
                proc = subprocess.Popen(
                    [redis_bin, redis_conf, "--port", str(self._ports[i])],
                    stdout=None if verbose > 2 else subprocess.DEVNULL,
                    stderr=None if verbose > 2 else subprocess.DEVNULL,
                    shell=False,
                )
                procs.append(proc)
                atexit.register(proc.terminate)
        self._timeout = timeout
        self._wait_interval = wait_interval
        self._verbose = verbose
        self._stats = RedisStoreStats()
        time.sleep(self._wait_interval)
        self._redis_clients = [None for _ in range(num_shards)]
        for i in range(num_shards):
            curr_time = time.time()
            if self._verbose > 0:
                print(f"Connecting to Redis server at {host}:{self._ports[i]}")
            self._redis_clients[i] = redis.Redis(
                host=host, port=self._ports[i], db=0, max_connections=max_connections
            )
            while True:
                try:
                    self._redis_clients[i].ping()
                    break
                except redis.exceptions.ConnectionError:
                    if self._verbose > 0:
                        print(
                            "Failed to ping Redis server at "
                            f"{host}:{self._ports[i]}, retrying..."
                        )
                    if time.time() - curr_time > self._timeout.total_seconds():
                        raise TimeoutError("Timeout connecting to Redis server")
                    time.sleep(0.1)
        if self._verbose > 0:
            print(f"Connected to {num_shards} Redis server")
        self._stats.startup_time = time.time() - start_time

    def get_shard_idx(self, key: str) -> int:
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % len(self._redis_clients)

    def reset_stats(self):
        self._stats.reset()

    def dump_stats(self):
        self._stats.dump()

    def set(self, key: str, value: str):
        start_time = time.time()
        shard_idx = self.get_shard_idx(key)
        if self._verbose > 0:
            print(f"Setting {key} shard={shard_idx} at {self._get_timestamp()}")
        self._redis_clients[shard_idx].set(key, value)
        self._stats.add_delay("set", time.time() - start_time)

    def get(self, key: str) -> bytes:
        start_time = time.time()
        shard_idx = self.get_shard_idx(key)
        if self._verbose > 0:
            print(f"Getting {key} shard={shard_idx} at {self._get_timestamp()}")
        start_time = time.time()
        retry = 0
        value = b""
        while True:
            value = self._redis_clients[shard_idx].get(key)
            if value is not None:
                break
            if time.time() - start_time > self._timeout.total_seconds():
                raise TimeoutError(f"Timeout getting key {key}")
            time.sleep(self._wait_interval)
            retry += 1
        self._stats.add_delay("get", time.time() - start_time, retry)
        if self._verbose > 1:
            print(f"Got {key} shard={shard_idx} at {self._get_timestamp()}")
        return value

    def add(self, key: str, value: int) -> int:
        start_time = time.time()
        res = self._redis_clients[self.get_shard_idx()].incrby(key, value)
        self._stats.add_delay("add", time.time() - start_time)
        return res

    def compare_set(
        self,
        key: str,
        expected_value: str,
        desired_value: str,
    ) -> bytes:
        shard_idx = self.get_shard_idx(key)
        start_time = time.time()
        existing_value = self._redis_clients[shard_idx].get(key)
        res = None
        if existing_value is None:
            if expected_value == "":
                self._redis_clients[shard_idx].set(key, desired_value)
                res = desired_value.encode()
            else:
                res = expected_value.encode()
        else:
            if existing_value == expected_value:
                self._redis_clients[shard_idx].set(key, desired_value)
            res = existing_value
        self._stats.add_delay("compare", time.time() - start_time)
        return res

    def delete_key(self, key: str) -> bool:
        start_time = time.time()
        res = self._redis_clients[self.get_shard_idx()].delete(key) > 0
        self._stats.add_delay("delete", time.time() - start_time)
        return res

    def num_keys(self) -> int:
        count = 0
        for i in range(len(self._redis_clients)):
            count += len(self._redis_clients[i].keys())
        return count

    def set_timeout(self, timeout: timedelta):
        self._timeout = timeout

    @overload
    def wait(self, keys: List[str]): ...

    @overload
    def wait(self, keys: List[str], timeout: timedelta): ...

    def wait(self, keys: List[str], timeout: Optional[timedelta] = None):
        start_time = time.time()
        retry = 0
        if self._verbose > 0:
            print(f"Waiting for {keys} at {self._get_timestamp()}")
        eff_timeout = self._timeout if timeout is None else timeout
        start_time = time.time()
        remaining_keys = keys
        while len(remaining_keys) > 0:
            current_keys = remaining_keys
            remaining_keys = []
            for key in current_keys:
                shard_idx = self.get_shard_idx(key)
                value = self._redis_clients[shard_idx].get(key)
                if value is None:
                    remaining_keys.append(key)
            if len(remaining_keys) == 0:
                break
            curr_time = time.time()
            if (curr_time - start_time) > eff_timeout.total_seconds():
                raise TimeoutError(f"Timeout waiting for keys {remaining_keys}")
            time.sleep(self._wait_interval)
            retry += 1
        self._stats.add_delay("wait", time.time() - start_time, retry)
