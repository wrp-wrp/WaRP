"""
Request Executor — Abstraction for I/O plan execution

Provides different execution strategies for issuing S3 range requests:
  - SequentialExecutor: issue requests one-by-one (baseline)
  - ConcurrentExecutor: issue requests in parallel via ThreadPoolExecutor
  - AsyncConcurrentExecutor: issue requests via asyncio + aiohttp (no GIL)
  - PoolExecutor: bounded concurrency pool

For concurrent executors, optional request scheduler:
  - fifo: preserve plan order
  - lpt: longest span first
  - auto: fifo for single-wave, lpt for multi-wave

Each executor takes an io_plan (list of (offset, length)) and returns
per-request latencies so the caller can aggregate them.
"""

from __future__ import annotations

import time
import asyncio
import logging
import math
import threading
import boto3
import botocore.config
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError:  # pragma: no cover - optional dependency for async executor
    aiohttp = None


_VALID_SCHEDULER_STRATEGIES = {"fifo", "lpt", "auto"}


def _normalize_scheduler_strategy(strategy: Optional[str]) -> str:
    """Normalize scheduler strategy name."""
    st = str(strategy or "auto").strip().lower()
    if st not in _VALID_SCHEDULER_STRATEGIES:
        logger.warning("Unknown scheduler strategy %r; falling back to 'auto'.", strategy)
        return "auto"
    return st


def _order_indexed_io_plan(
    io_plan: List[Tuple[int, int]],
    scheduler_strategy: Optional[str],
    max_workers: int,
) -> List[Tuple[int, int, int]]:
    """Return [(orig_idx, offset, length)] ordered by scheduler policy."""
    ordered = [(idx, int(offset), int(length)) for idx, (offset, length) in enumerate(io_plan)]
    if not ordered:
        return ordered

    strategy = _normalize_scheduler_strategy(scheduler_strategy)
    if strategy == "auto":
        # Auto uses LPT only when requests must run in multiple waves.
        strategy = "lpt" if len(ordered) > max(1, int(max_workers)) else "fifo"

    if strategy == "lpt":
        ordered.sort(key=lambda x: (-x[2], x[0]))
    return ordered


class _TokenBucketRateLimiter:
    """Simple thread-safe token bucket for request-rate limiting."""

    def __init__(self, rate_per_sec: float, burst_seconds: float = 1.0):
        self._rate_per_sec = max(1e-6, float(rate_per_sec))
        burst_seconds = max(1e-3, float(burst_seconds))
        self._capacity = max(1.0, self._rate_per_sec * burst_seconds)
        self._tokens = float(self._capacity)
        self._last_refill_s = time.monotonic()
        self._cv = threading.Condition()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill_s
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
        self._last_refill_s = now

    def acquire(self, tokens: float = 1.0) -> None:
        need = max(1e-6, float(tokens))
        with self._cv:
            while True:
                self._refill_locked()
                if self._tokens >= need:
                    self._tokens -= need
                    return
                wait_s = (need - self._tokens) / self._rate_per_sec
                self._cv.wait(timeout=max(1e-4, wait_s))


def _build_rate_limiter(
    max_request_rate: Optional[float],
    rate_limit_burst: Optional[float],
) -> Optional[_TokenBucketRateLimiter]:
    if max_request_rate is None:
        return None
    try:
        rate = float(max_request_rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    burst = 1.0
    if rate_limit_burst is not None:
        try:
            burst = float(rate_limit_burst)
        except (TypeError, ValueError):
            burst = 1.0
    if not math.isfinite(burst) or burst <= 0.0:
        burst = 1.0
    return _TokenBucketRateLimiter(rate_per_sec=rate, burst_seconds=burst)


class ExecutionResult:
    """Result of executing an I/O plan."""

    def __init__(self, per_request_latencies: List[float], total_latency_ms: float,
                 total_bytes: int):
        self.per_request_latencies = per_request_latencies  # ms each
        self.total_latency_ms = total_latency_ms
        self.total_bytes = total_bytes
        self.num_requests = len(per_request_latencies)

    def __repr__(self):
        return (f"ExecutionResult(requests={self.num_requests}, "
                f"total={self.total_latency_ms:.1f}ms, "
                f"bytes={self.total_bytes})")


class RequestExecutor(ABC):
    """Abstract base class for request execution strategies."""

    def __init__(
        self,
        max_request_rate: Optional[float] = None,
        rate_limit_burst: Optional[float] = 1.0,
    ):
        self._rate_limiter = _build_rate_limiter(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
        self.max_request_rate = (
            float(max_request_rate)
            if isinstance(max_request_rate, (int, float))
            else None
        )
        self.rate_limit_burst = (
            float(rate_limit_burst)
            if isinstance(rate_limit_burst, (int, float))
            else None
        )

    @abstractmethod
    def execute(self, io_plan: List[Tuple[int, int]], s3_client, bucket: str,
                key: str) -> ExecutionResult:
        """
        Execute an I/O plan against S3.

        Args:
            io_plan: List of (offset, length) for range requests
            s3_client: boto3 S3 client
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            ExecutionResult with timing data
        """
        pass

    def _throttle_request(self) -> None:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

    async def _athrottle_request(self) -> None:
        if self._rate_limiter is not None:
            await asyncio.to_thread(self._rate_limiter.acquire)

    def close(self) -> None:
        """Release any resources held by the executor."""
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _do_range_request(s3_client, bucket: str, key: str,
                          offset: int, length: int) -> Tuple[bytes, float]:
        """
        Issue a single S3 range request, return (data, latency_ms).
        """
        range_header = f'bytes={offset}-{offset + length - 1}'
        t0 = time.time()
        resp = s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
        data = resp['Body'].read()
        latency_ms = (time.time() - t0) * 1000
        return data, latency_ms

    def _do_range_request_with_rate_limit(
        self, s3_client, bucket: str, key: str, offset: int, length: int
    ) -> Tuple[bytes, float]:
        self._throttle_request()
        return self._do_range_request(s3_client, bucket, key, offset, length)


class SequentialExecutor(RequestExecutor):
    """Execute requests sequentially (one at a time)."""

    def __init__(
        self,
        max_request_rate: Optional[float] = None,
        rate_limit_burst: Optional[float] = 1.0,
    ):
        super().__init__(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )

    def execute(self, io_plan, s3_client, bucket, key):
        latencies = []
        total_bytes = 0
        t0 = time.time()

        for offset, length in io_plan:
            _, lat = self._do_range_request_with_rate_limit(
                s3_client, bucket, key, offset, length
            )
            latencies.append(lat)
            total_bytes += length

        total_ms = (time.time() - t0) * 1000
        return ExecutionResult(latencies, total_ms, total_bytes)


class ConcurrentExecutor(RequestExecutor):
    """Execute all requests concurrently via ThreadPoolExecutor.

    Uses the provided S3 client. Ensure its urllib3 connection pool
    (max_pool_connections) is sized to at least max_workers.
    """

    def __init__(
        self,
        max_workers: int = 32,
        region: str = 'ap-northeast-1',
        scheduler_strategy: str = 'auto',
        max_request_rate: Optional[float] = None,
        rate_limit_burst: Optional[float] = 1.0,
    ):
        super().__init__(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
        self.max_workers = max_workers
        self._fallback_region = region
        self._fallback_client = None
        self.scheduler_strategy = _normalize_scheduler_strategy(scheduler_strategy)
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(
            "ConcurrentExecutor: max_workers=%s, scheduler=%s",
            max_workers,
            self.scheduler_strategy,
        )

    def execute(self, io_plan, s3_client, bucket, key):
        if not io_plan:
            return ExecutionResult([], 0.0, 0)

        client = s3_client
        if client is None:
            if self._fallback_client is None:
                cfg = botocore.config.Config(
                    max_pool_connections=self.max_workers,
                )
                self._fallback_client = boto3.client(
                    's3', region_name=self._fallback_region, config=cfg
                )
            client = self._fallback_client
        total_bytes = sum(length for _, length in io_plan)
        t0 = time.time()

        ordered_plan = _order_indexed_io_plan(
            io_plan,
            scheduler_strategy=self.scheduler_strategy,
            max_workers=self.max_workers,
        )
        latencies = [0.0] * len(io_plan)
        pool = self._pool
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=self.max_workers)
            self._pool = pool
        futures: Dict = {}
        max_inflight = max(1, min(self.max_workers, len(ordered_plan)))
        submit_cursor = 0

        def _submit_one(orig_idx: int, offset: int, length: int) -> None:
            fut = pool.submit(
                self._do_range_request_with_rate_limit,
                client,
                bucket,
                key,
                offset,
                length,
            )
            futures[fut] = orig_idx

        while submit_cursor < max_inflight:
            orig_idx, offset, length = ordered_plan[submit_cursor]
            _submit_one(orig_idx, offset, length)
            submit_cursor += 1

        while futures:
            done, _pending = wait(tuple(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                idx = futures.pop(fut)
                _, lat = fut.result()
                latencies[idx] = lat
                if submit_cursor < len(ordered_plan):
                    orig_idx, offset, length = ordered_plan[submit_cursor]
                    _submit_one(orig_idx, offset, length)
                    submit_cursor += 1

        total_ms = (time.time() - t0) * 1000
        return ExecutionResult(latencies, total_ms, total_bytes)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


class PoolExecutor(RequestExecutor):
    """Execute requests with a fixed-size connection pool (bounded concurrency)."""

    def __init__(
        self,
        pool_size: int = 5,
        max_request_rate: Optional[float] = None,
        rate_limit_burst: Optional[float] = 1.0,
    ):
        super().__init__(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
        self.pool_size = pool_size
        self._pool = ThreadPoolExecutor(max_workers=self.pool_size)

    def execute(self, io_plan, s3_client, bucket, key):
        total_bytes = sum(length for _, length in io_plan)
        t0 = time.time()

        latencies = [0.0] * len(io_plan)
        pool = self._pool
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=self.pool_size)
            self._pool = pool
        futures = {}
        for idx, (offset, length) in enumerate(io_plan):
            fut = pool.submit(
                self._do_range_request_with_rate_limit,
                s3_client,
                bucket,
                key,
                offset,
                length,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            _, lat = fut.result()
            latencies[idx] = lat

        total_ms = (time.time() - t0) * 1000
        return ExecutionResult(latencies, total_ms, total_bytes)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


class AsyncConcurrentExecutor(RequestExecutor):
    """Execute all requests concurrently via asyncio + aiohttp.

    Eliminates ThreadPoolExecutor dispatch overhead and GIL contention
    by using native async I/O.  Generates pre-signed S3 URLs (sync, fast)
    then fires all requests through a single aiohttp.ClientSession with a
    TCPConnector sized to max_workers.
    """

    # Presigned URL TTL — 5 minutes is more than enough for a query batch
    _PRESIGN_EXPIRY = 300

    def __init__(
        self,
        max_workers: int = 100,
        region: str = 'ap-northeast-1',
        scheduler_strategy: str = 'auto',
        max_request_rate: Optional[float] = None,
        rate_limit_burst: Optional[float] = 1.0,
    ):
        super().__init__(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
        if aiohttp is None:
            raise ImportError(
                "aiohttp is required for AsyncConcurrentExecutor. Install dependencies from requirements.txt."
            )
        self.max_workers = max_workers
        self.region = region
        self.scheduler_strategy = _normalize_scheduler_strategy(scheduler_strategy)
        cfg = botocore.config.Config(
            max_pool_connections=max_workers,
            signature_version='s3v4',
        )
        self._s3_client = boto3.client('s3', region_name=region, config=cfg)
        self._runner = asyncio.Runner()
        self._session = None
        self._presigned_cache = {}
        logger.info(
            "AsyncConcurrentExecutor: max_workers=%s, scheduler=%s",
            max_workers,
            self.scheduler_strategy,
        )

    # ---- public API (sync wrapper) ----

    def execute(self, io_plan, s3_client, bucket, key):
        """Synchronous entry-point: runs the async loop internally."""
        return self._runner.run(self._execute_async(io_plan, bucket, key))

    def close(self) -> None:
        try:
            if self._session is not None:
                self._runner.run(self._session.close())
                self._session = None
        finally:
            self._runner.close()

    # ---- async internals ----

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_workers,
                limit_per_host=self.max_workers,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    def _get_presigned_url(self, bucket: str, key: str) -> str:
        cache_key = (bucket, key)
        now = time.time()
        entry = self._presigned_cache.get(cache_key)
        if entry is not None:
            url, expires_at = entry
            if now < expires_at:
                return url
        url = self._s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=self._PRESIGN_EXPIRY,
        )
        self._presigned_cache[cache_key] = (url, now + self._PRESIGN_EXPIRY - 5)
        return url

    async def _execute_async(self, io_plan, bucket, key):
        if not io_plan:
            return ExecutionResult([], 0.0, 0)

        total_bytes = sum(length for _, length in io_plan)
        ordered_plan = _order_indexed_io_plan(
            io_plan,
            scheduler_strategy=self.scheduler_strategy,
            max_workers=self.max_workers,
        )

        url = self._get_presigned_url(bucket, key)
        session = await self._get_session()

        latencies = [0.0] * len(io_plan)
        t0 = time.time()

        sem = asyncio.Semaphore(max(1, min(self.max_workers, len(ordered_plan))))

        async def _fetch_limited(idx: int, offset: int, length: int) -> Tuple[int, float]:
            range_header = f'bytes={offset}-{offset + length - 1}'
            async with sem:
                await self._athrottle_request()
                return await self._fetch_one(session, idx, url, range_header)

        tasks = []
        for idx, offset, length in ordered_plan:
            tasks.append(_fetch_limited(idx, offset, length))
        results = await asyncio.gather(*tasks)

        for idx, lat in results:
            latencies[idx] = lat

        total_ms = (time.time() - t0) * 1000
        return ExecutionResult(latencies, total_ms, total_bytes)

    @staticmethod
    async def _fetch_one(session: aiohttp.ClientSession, idx: int,
                         url: str, range_header: str) -> Tuple[int, float]:
        """Fetch a single range request, return (index, latency_ms)."""
        t0 = time.time()
        async with session.get(url, headers={'Range': range_header}) as resp:
            resp.raise_for_status()
            await resp.read()
        latency_ms = (time.time() - t0) * 1000
        return idx, latency_ms


def create_executor(executor_type: str = 'sequential', **kwargs) -> RequestExecutor:
    """
    Factory function to create a RequestExecutor.

    Args:
        executor_type: 'sequential', 'concurrent', 'async_concurrent', or 'pool'
        **kwargs: Additional arguments (e.g., max_workers, pool_size, region,
            scheduler_strategy for concurrent executors)

    Returns:
        RequestExecutor instance
    """
    max_request_rate = kwargs.get('max_request_rate')
    rate_limit_burst = kwargs.get('rate_limit_burst', 1.0)
    if executor_type == 'sequential':
        return SequentialExecutor(
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
    elif executor_type == 'concurrent':
        return ConcurrentExecutor(
            max_workers=kwargs.get('max_workers', 32),
            region=kwargs.get('region', 'ap-northeast-1'),
            scheduler_strategy=kwargs.get('scheduler_strategy', 'auto'),
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
    elif executor_type == 'async_concurrent':
        return AsyncConcurrentExecutor(
            max_workers=kwargs.get('max_workers', 100),
            region=kwargs.get('region', 'ap-northeast-1'),
            scheduler_strategy=kwargs.get('scheduler_strategy', 'auto'),
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
    elif executor_type == 'pool':
        return PoolExecutor(
            pool_size=kwargs.get('pool_size', 5),
            max_request_rate=max_request_rate,
            rate_limit_burst=rate_limit_burst,
        )
    else:
        raise ValueError(f"Unknown executor type: {executor_type}. "
                         f"Supported: sequential, concurrent, async_concurrent, pool")
