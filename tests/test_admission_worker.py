"""Backpressure and cancellation must not abandon an accepted durable write."""

import asyncio
import threading

import pytest

from eugene_plexus_control.admission_worker import AdmissionWorker


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_its_place_until_the_write_commits():
    worker = AdmissionWorker(capacity=1)
    entered = threading.Event()
    release = threading.Event()
    committed = []

    def write():
        entered.set()
        assert release.wait(5)
        committed.append("durable")

    request = asyncio.create_task(worker.run(write))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(OSError, match="busy"):
            await worker.run(lambda: None)
        assert not committed
    finally:
        release.set()
        await worker.close()
    assert committed == ["durable"]
    with pytest.raises(OSError):
        await worker.run(lambda: None)
