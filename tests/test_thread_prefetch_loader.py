import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader

from pretrain_src.pretrain_src.data import loader as loader_module
from pretrain_src.pretrain_src.data.loader import MetaLoader, ThreadPrefetchLoader


class _FiniteLoader:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        for value in self.values:
            yield {"value": torch.tensor([value])}

    def __len__(self):
        return len(self.values)


class _ScheduledLoader:
    def __init__(self):
        self.next_value = 0
        self.reserve_threads = []
        self.load_threads = []

    def reserve_next_task(self):
        self.reserve_threads.append(threading.get_ident())
        value = self.next_value
        self.next_value += 1
        return value

    def load_batch_for_task(self, value):
        self.load_threads.append(threading.get_ident())
        return {"value": torch.tensor([value])}


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


def test_scheduled_prefetch_keeps_task_selection_on_consumer_thread():
    source = _ScheduledLoader()
    main_thread = threading.get_ident()
    iterator = iter(
        ThreadPrefetchLoader(
            source,
            torch.device("cpu"),
            prefetch_size=1,
        )
    )

    values = [int(next(iterator)["value"][0]) for _ in range(3)]
    iterator.close()

    assert values == [0, 1, 2]
    assert source.reserve_threads
    assert set(source.reserve_threads) == {main_thread}
    assert source.load_threads
    assert main_thread not in set(source.load_threads)


def test_distributed_task_broadcast_stays_on_consumer_thread(monkeypatch):
    main_thread = threading.get_ident()
    broadcast_threads = []
    monkeypatch.setattr(
        loader_module.dist,
        "broadcast",
        lambda _task_id, _source: broadcast_threads.append(
            threading.get_ident()
        ),
    )
    task_loader = DataLoader(torch.arange(4), batch_size=1)
    meta_loader = MetaLoader(
        {"mlm": (task_loader, None, lambda _epoch: None)},
        {"-1": [1.0]},
        accum_steps=1,
        distributed=True,
        device=torch.device("cpu"),
    )
    iterator = iter(
        ThreadPrefetchLoader(
            meta_loader,
            torch.device("cpu"),
            prefetch_size=1,
        )
    )

    task, batch = next(iterator)
    iterator.close()

    assert task == "mlm"
    assert batch.tolist() == [0]
    assert broadcast_threads
    assert set(broadcast_threads) == {main_thread}
