#!/usr/bin/env python3
"""Control two bounded CUDA captures; run only in the profiling deployment."""

import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path('/experiment')
engine = json.loads((ROOT / 'engine.json').read_text())
deployment = json.loads((ROOT / 'deployment.json').read_text())
benchmark = json.loads((ROOT / 'benchmark.json').read_text())
url = f"http://127.0.0.1:{engine['network']['port']}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + deployment['startup_seconds'] + benchmark['timeout_seconds']


def request(path, post=False):
    req = urllib.request.Request(url + path, data=b'{}' if post else None,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=60) as response:
        return response.read().decode()


def wait_until(predicate):
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1)
    raise TimeoutError('Timed out waiting for profiling workload')


def healthy():
    try:
        request('/health')
        return True
    except (urllib.error.URLError, TimeoutError):
        return False


def metrics():
    result = {}
    for line in request('/metrics').splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in ('vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
                    'vllm:num_requests_running'):
            result[name] = result.get(name, 0) + float(line.split()[-1])
    return result


def measurement_started():
    path = ROOT / 'benchmark.log'
    if not path.exists():
        return False
    log = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text(errors='replace'))
    return bool(re.search(r'Phase .*\(profiling\) started', log, re.I))


def capture(label, minimum_generated, wait_for_prompt):
    before = metrics()
    started = time.time()
    print(f'Starting {label}: {before}', flush=True)
    request('/start_profile', post=True)
    try:
        def enough_tokens():
            now = metrics()
            generated = now.get('vllm:generation_tokens_total', 0) - before.get('vllm:generation_tokens_total', 0)
            prompt = now.get('vllm:prompt_tokens_total', 0) - before.get('vllm:prompt_tokens_total', 0)
            return generated >= minimum_generated and (not wait_for_prompt or prompt > 0)
        wait_until(enough_tokens)
        time.sleep(2)
    finally:
        request('/stop_profile', post=True)
    record = {'label': label, 'start_unix': started, 'stop_unix': time.time(),
              'metrics_before': before, 'metrics_after': metrics()}
    (ROOT / 'nsys' / f'{label}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Finished {label}: {record}', flush=True)


wait_until(healthy)
# AgentX warmup uses max_tokens=1. Capture its first large prefill separately
# from measured decoding. CPU sampling is disabled during the preceding idle.
capture('warmup-prefill', 1, True)
wait_until(measurement_started)
wait_until(lambda: metrics().get('vllm:num_requests_running', 0) > 0)
capture('after-warmup', 128, False)
(ROOT / 'nsys' / 'completed.json').write_text('{"captures": 2}\n')
