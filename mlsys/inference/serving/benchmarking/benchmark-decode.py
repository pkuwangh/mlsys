#!/usr/bin/env python3
"""Sequential raw-completion decode measurements, with timestamped token IDs."""

import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request


def measure(url, payload, path, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url + '/v1/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    token_chunks, usage, finish_reason, done = [], None, None, False
    started = time.perf_counter()
    with path.open('w') as saved, opener.open(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            elapsed = time.perf_counter() - started
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            saved.write(json.dumps({'elapsed_seconds': elapsed, 'event': event}) + '\n')
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('token_ids'):
                    token_chunks.append((elapsed, len(choice['token_ids'])))
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']
    duration = time.perf_counter() - started
    count = sum(n for _, n in token_chunks)
    if not done or finish_reason != 'length' or not usage:
        raise RuntimeError(f'Incomplete response: done={done}, finish={finish_reason}, usage={usage}')
    if count != payload['max_tokens'] or usage['completion_tokens'] != count:
        raise RuntimeError(f'Token count mismatch: stream={count}, usage={usage}')
    if usage['prompt_tokens'] != len(payload['prompt']) or len(token_chunks) < 2:
        raise RuntimeError(f'Unexpected prompt length or non-streaming response: {usage}, chunks={len(token_chunks)}')
    first, first_count = token_chunks[0]
    last = token_chunks[-1][0]
    decode_tokens = count - first_count
    gaps = [b[0] - a[0] for a, b in zip(token_chunks, token_chunks[1:])]
    return {'output_tokens': count, 'prompt_tokens': usage['prompt_tokens'],
            'first_chunk_tokens': first_count, 'token_chunks': len(token_chunks),
            'ttft_seconds': first, 'request_seconds': duration,
            'decode_seconds': last - first, 'full_decode_seconds': duration - first,
            'decode_tokens_per_second': decode_tokens / (last - first),
            'full_response_tokens_per_second': decode_tokens / (duration - first),
            'median_chunk_interval_ms': statistics.median(gaps) * 1000,
            'mean_tokens_per_chunk_after_first': decode_tokens / (len(token_chunks) - 1),
            'finish_reason': finish_reason}


def main():
    root = Path('/experiment')
    engine = json.loads((root / 'engine.json').read_text())
    bench = json.loads((root / 'benchmark.json').read_text())
    url = f"http://{sys.argv[1]}:{engine['network']['port']}"
    output = root / 'decode'
    output.mkdir(exist_ok=True)
    payload = {'model': engine['model']['served_name'], 'prompt': bench['prompt_token_ids'],
               'max_tokens': bench['output_tokens'], 'temperature': 0, 'seed': bench['seed'],
               'ignore_eos': True, 'stream': True, 'return_token_ids': True,
               'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
    (output / 'request.json').write_text(json.dumps(payload, indent=2) + '\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save_metrics(name):
        with opener.open(url + '/metrics', timeout=30) as response:
            (output / name).write_bytes(response.read())

    for i in range(bench['warmup_requests']):
        result = measure(url, payload, output / f'warmup-{i}.jsonl', bench['request_timeout_seconds'])
        print(json.dumps({'phase': 'warmup', 'request': i, **result}), flush=True)
    save_metrics('metrics-before.txt')
    repeats = []
    with (output / 'requests.jsonl').open('w') as records:
        for repeat in range(bench['repeats']):
            results = []
            for i in range(bench['requests_per_repeat']):
                result = measure(url, payload, output / f'repeat-{repeat}-request-{i}.jsonl', bench['request_timeout_seconds'])
                record = {'repeat': repeat, 'request': i, **result}
                records.write(json.dumps(record) + '\n')
                records.flush()
                print(json.dumps(record), flush=True)
                results.append(result)
            repeats.append({'repeat': repeat,
                            'median_decode_tokens_per_second': statistics.median(r['decode_tokens_per_second'] for r in results),
                            'median_full_response_tokens_per_second': statistics.median(r['full_response_tokens_per_second'] for r in results)})
    save_metrics('metrics-after.txt')
    median = statistics.median(r['median_decode_tokens_per_second'] for r in repeats)
    summary = {'benchmark': bench, 'repeats': repeats,
               'median_decode_tokens_per_second': median,
               'published_blog_tokens_per_second': 842,
               'published_pr_tokens_per_second': 838,
               'difference_from_blog_percent': 100 * (median / 842 - 1),
               'note': 'Client-observed token-ID delivery, excluding the first chunk. Published PR uses decode steps/s times acceptance length. Historical image and short-context DCP setting are not pinned.'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
