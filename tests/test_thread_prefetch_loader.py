import threading
import time

import pytest
import torch

from pretrain_src.pretrain_src.data.loader import ThreadPrefetchLoader


class _FiniteLoader:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        for value in self.values:
            yield {"value": torch.tensor([value])}

    def __len__(self):
        return len(self.values)


def test_thread_prefetch_loader_preserves_order_and_moves_nested_tensors():
    loader = ThreadPrefetchLoader(
        _FiniteLoader([1, 2, 3]),
        torch.device("cpu"),
        prefetch_size=1,
    )

    values = [int(batch["value"][0]) for batch in loader]

    assert values == [1, 2, 3]
    assert len(loader) == 3


def test_thread_prefetch_loader_propagates_producer_exception():
    class FailingLoader:
        def __iter__(self):
            yield {"value": torch.tensor([1])}
            raise ValueError("broken batch")

    iterator = iter(
        ThreadPrefetchLoader(FailingLoader(), torch.device("cpu"))
    )

    assert int(next(iterator)["value"][0]) == 1
    with pytest.raises(RuntimeError, match="broken batch") as caught:
        next(iterator)
    assert isinstance(caught.value.__cause__, ValueError)


def test_thread_prefetch_loader_stops_daemon_thread_when_consumer_closes():
    class EndlessLoader:
        def __iter__(self):
            value = 0
            while True:
                yield {"value": torch.tensor([value])}
                value += 1

    iterator = iter(
        ThreadPrefetchLoader(
            EndlessLoader(),
            torch.device("cpu"),
            prefetch_size=1,
        )
    )
    next(iterator)
    iterator.close()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not any(
            thread.name == "etpr1-cpu-batch-prefetch"
            for thread in threading.enumerate()
        ):
            break
        time.sleep(0.01)

    assert not any(
        thread.name == "etpr1-cpu-batch-prefetch"
        for thread in threading.enumerate()
    )


def test_thread_prefetch_loader_rejects_empty_queue():
    with pytest.raises(ValueError, match="at least 1"):
        ThreadPrefetchLoader(
            _FiniteLoader([]),
            torch.device("cpu"),
            prefetch_size=0,
        )
