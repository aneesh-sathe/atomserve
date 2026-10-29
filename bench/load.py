# HTTP load generator: how latency and throughput change as more requests share the engine.
#
# For each concurrency level C, C client threads keep sending streaming chat
# requests (greedy, fixed output length via ignore_eos) until the level's
# request count is done. Reported per level:
#
#   throughput  generated tokens per second across all requests (what batching buys)
#   TTFT        time to first token: queueing + prefill
#   ITL         inter-token latency: the gap between streamed chunks of one request
#
# `--long-prompt-tokens N` also sends one request with an N-token prompt while
# the short streams run, which shows how a big prefill stalls everyone else
# (and how chunked prefill bounds that stall).
#
# Usage (with `atomserve Qwen/Qwen3-0.6B` running):
#   uv run python bench/load.py --concurrency 1,4,16,32
#   uv run python bench/load.py --concurrency 8 --long-prompt-tokens 8000

import argparse
import http.client
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."
FILLER = "The quick brown fox jumps over the lazy dog. "  # about 10 tokens


@dataclass
class Result:
    ttft: float = 0.0  # seconds to the first content chunk
    chunk_times: list[float] = field(default_factory=list)  # arrival time of every chunk
    completion_tokens: int = 0
    prompt_tokens: int = 0
    text: str = ""


def stream_chat(url: str, prompt: str, max_tokens: int) -> Result:
    """Send one streaming chat request and time every chunk of the reply."""
    target = urlparse(url)
    conn = http.client.HTTPConnection(target.hostname, target.port, timeout=600)
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result, start = Result(), time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body))
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if usage := chunk.get("usage"):
            result.completion_tokens = usage["completion_tokens"]
            result.prompt_tokens = usage["prompt_tokens"]
        for choice in chunk["choices"]:
            if text := choice["delta"].get("content"):
                now = time.perf_counter()
                if not result.chunk_times:
                    result.ttft = now - start
                result.chunk_times.append(now)
                result.text += text
    conn.close()
    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


def run_level(args: argparse.Namespace, concurrency: int) -> None:
    """Run one concurrency level and print its row."""
    num_requests = args.requests or max(4, 2 * concurrency)
    results: list[Result] = []
    lock = threading.Lock()
    remaining = [num_requests]

    def client() -> None:
        while True:
            with lock:
                if remaining[0] == 0:
                    return
                remaining[0] -= 1
            result = stream_chat(args.url, PROMPT, args.max_tokens)
            with lock:
                results.append(result)

    long_result: list[Result] = []

    def long_client() -> None:
        time.sleep(1.0)  # let the short streams get going first
        prompt = FILLER * (args.long_prompt_tokens // 10) + "\nSummarize the text above."
        long_result.append(stream_chat(args.url, prompt, 1))

    start = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(concurrency)]
    if args.long_prompt_tokens:
        threads.append(threading.Thread(target=long_client))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start

    tokens = sum(r.completion_tokens for r in results)
    itls = [b - a for r in results for a, b in zip(r.chunk_times, r.chunk_times[1:], strict=False)]
    ttfts = [r.ttft for r in results]
    distinct = len({r.text for r in results})
    row = (
        f"{concurrency:>5} {len(results):>5} {tokens / wall:>10.1f} "
        f"{statistics.median(ttfts) * 1000:>9.0f} {percentile(ttfts, 99) * 1000:>9.0f} "
        f"{statistics.median(itls) * 1000:>8.1f} {percentile(itls, 99) * 1000:>8.1f} "
        f"{max(itls) * 1000:>8.1f} {distinct:>9}"
    )
    print(row, flush=True)
    if long_result:
        r = long_result[0]
        print(f"      long prompt: {r.prompt_tokens} tokens, TTFT {r.ttft * 1000:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test an atomserve (OpenAI) endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:11435")
    parser.add_argument("--concurrency", default="1,4,16,32", help="comma-separated levels")
    parser.add_argument("--requests", type=int, default=0, help="per level (default 2x C, min 4)")
    parser.add_argument("--max-tokens", type=int, default=128, help="tokens generated per request")
    parser.add_argument("--long-prompt-tokens", type=int, default=0)
    args = parser.parse_args()

    stream_chat(args.url, "Hi", 4)  # warm-up
    print(
        f"{'conc':>5} {'reqs':>5} {'tok/s':>10} {'TTFT p50':>9} {'TTFT p99':>9} "
        f"{'ITL p50':>8} {'ITL p99':>8} {'ITL max':>8} {'distinct':>9}"
    )
    print(f"{'':>5} {'':>5} {'':>10} {'(ms)':>9} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8}")
    for level in args.concurrency.split(","):
        run_level(args, int(level))


if __name__ == "__main__":
    main()
