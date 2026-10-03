"""One bounded worker owns admission decisions and their durable commits."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class AdmissionWorker:
    def __init__(self, *, capacity: int = 256) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="control-admission")
        self._capacity = capacity
        self._pending = 0
        self._closed = False

    async def run[T](self, operation: Callable[[], T]) -> T:
        if self._closed or self._pending >= self._capacity:
            raise OSError("Admission writer is busy; retry later.")
        self._pending += 1
        future = asyncio.get_running_loop().run_in_executor(self._executor, operation)

        def finished(done: asyncio.Future[T]) -> None:
            self._pending -= 1
            # A disconnected caller does not abandon a commit already queued.
            # Retrieve failures even if that caller is no longer waiting.
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True)
