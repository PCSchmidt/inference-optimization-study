"""Adaptive batching with a request queue for throughput optimization."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BatchRequest:
    text: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.perf_counter)


class AdaptiveBatcher:
    """Collects individual requests into batches for efficient inference.

    Parameters
    ----------
    encode_fn : callable
        Function that takes a list of strings and returns np.ndarray of embeddings.
    max_batch_size : int
        Maximum number of requests per batch.
    max_wait_ms : float
        Maximum time to wait (ms) for a batch to fill before dispatching.
    """

    def __init__(
        self,
        encode_fn,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ):
        self.encode_fn = encode_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._dispatch_task: asyncio.Task | None = None

    async def submit(self, text: str) -> np.ndarray:
        """Submit a single text and await its embedding."""
        req = BatchRequest(text=text)
        async with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.max_batch_size:
                await self._dispatch()
            elif self._dispatch_task is None or self._dispatch_task.done():
                self._dispatch_task = asyncio.create_task(self._wait_and_dispatch())
        return await req.future

    async def _wait_and_dispatch(self):
        """Wait up to max_wait_ms then dispatch whatever is queued."""
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        async with self._lock:
            if self._queue:
                await self._dispatch()

    async def _dispatch(self):
        """Process current queue as a batch."""
        batch = self._queue[: self.max_batch_size]
        self._queue = self._queue[self.max_batch_size :]
        texts = [r.text for r in batch]
        embeddings = self.encode_fn(texts)
        for req, emb in zip(batch, embeddings):
            if not req.future.done():
                req.future.set_result(emb)


def simulate_batch_throughput(
    encode_fn,
    texts: list[str],
    batch_sizes: list[int] = (1, 4, 8, 16, 32, 64),
) -> list[dict]:
    """Synchronous batch-size sweep for benchmarking.

    Returns list of dicts with batch_size, total_time_s, throughput_rps.
    """
    results = []
    for bs in batch_sizes:
        start = time.perf_counter()
        for i in range(0, len(texts), bs):
            encode_fn(texts[i : i + bs])
        elapsed = time.perf_counter() - start
        results.append({
            "batch_size": bs,
            "total_time_s": round(elapsed, 4),
            "throughput_rps": round(len(texts) / elapsed, 1),
        })
    return results
