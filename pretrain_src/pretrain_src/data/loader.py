"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

A prefetch loader to speedup data loading
Modified from Nvidia Deep Learning Examples
(https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
"""
import random
import queue
import threading
import traceback
from typing import List, Dict, Tuple, Union, Iterator

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import bisect

class MetaLoader:
    """wraps multiple data loaders"""

    def __init__(
        self, loaders, ratio_dict, accum_steps: int = 1, distributed: bool = False, device=None
    ):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2pre_epoch = {}
        self.names: List[str] = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r, p = l
            elif isinstance(l, DataLoader):
                r = 1
                p = lambda e: None
            else:
                raise ValueError()
            self.names.append(n)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2pre_epoch[n] = p

        self.accum_steps = accum_steps
        self.device = device
        self.sorted_iters = sorted(int(k) for k in ratio_dict)
        self.ratio_list = [torch.tensor(ratio_dict[str(k)]).float().to(device) for k in self.sorted_iters]
        self.distributed = distributed
        self.step = 0
        self._task_id = None
        self._epoch_id = 0

    def get_ratios(self, step):
        index = bisect.bisect_right(self.sorted_iters, step) - 1
        index = max(index, 0) 
        return self.ratio_list[index]
    
    def reserve_next_task(self):
        """Choose the next task on the training thread.

        Distributed task synchronization must stay on the same thread as DDP
        forward/backward collectives.  Batch materialization can then happen in
        a CPU-only producer thread after this method returns the task name.
        """
        if self.step % self.accum_steps == 0:
            update_step = self.step // self.accum_steps
            sampling_ratios = self.get_ratios(update_step)
            task_id = torch.multinomial(sampling_ratios, 1)
            if self.distributed:
                dist.broadcast(task_id, 0)
            self._task_id = int(task_id.item())
        elif self._task_id is None:
            raise RuntimeError(
                "MetaLoader task state is missing inside an accumulation step"
            )

        self.step += 1
        return self.names[self._task_id]

    def load_batch_for_task(self, task):
        """Materialize one CPU batch for a task without distributed calls."""
        iter_ = self.name2iter[task]
        try:
            batch = next(iter_)
        except StopIteration:
            self._epoch_id += 1
            self.name2pre_epoch[task](self._epoch_id)
            iter_ = iter(self.name2loader[task])
            batch = next(iter_)
            self.name2iter[task] = iter_
        return task, batch

    def __iter__(self) -> Iterator[Tuple]:
        """This iterator will run indefinitely."""
        while True:
            task = self.reserve_next_task()
            yield self.load_batch_for_task(task)


def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True) 
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


class _PrefetchException:
    def __init__(self, exception, formatted_traceback):
        self.exception = exception
        self.formatted_traceback = formatted_traceback


_PREFETCH_END = object()


class ThreadPrefetchLoader:
    """Prepare CPU batches in one background thread, without forking."""

    def __init__(self, loader, device: torch.device, prefetch_size: int = 2):
        if prefetch_size < 1:
            raise ValueError("prefetch_size must be at least 1")
        self.loader = loader
        self.device = device
        self.prefetch_size = prefetch_size

    def __iter__(self):
        if all(
            hasattr(self.loader, method)
            for method in ("reserve_next_task", "load_batch_for_task")
        ):
            yield from self._iter_scheduled_loader()
            return

        yield from self._iter_plain_loader()

    @staticmethod
    def _put_unless_stopped(target_queue, item, stop):
        while not stop.is_set():
            try:
                target_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _iter_plain_loader(self):
        source = iter(self.loader)
        batches = queue.Queue(maxsize=self.prefetch_size)
        stop = threading.Event()

        def put_unless_stopped(item):
            return self._put_unless_stopped(batches, item, stop)

        def produce():
            try:
                while not stop.is_set():
                    put_unless_stopped(next(source))
            except StopIteration:
                put_unless_stopped(_PREFETCH_END)
            except BaseException as exc:
                put_unless_stopped(
                    _PrefetchException(exc, traceback.format_exc())
                )

        producer = threading.Thread(
            target=produce,
            name="etpr1-cpu-batch-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = batches.get()
                if item is _PREFETCH_END:
                    return
                if isinstance(item, _PrefetchException):
                    raise RuntimeError(
                        "CPU batch prefetch failed:\n"
                        f"{item.formatted_traceback}"
                    ) from item.exception
                yield move_to_cuda(item, self.device)
        finally:
            stop.set()
            producer.join(timeout=1)

    def _iter_scheduled_loader(self):
        """Prefetch CPU batches while keeping task/NCCL work on this thread."""
        requests = queue.Queue(maxsize=self.prefetch_size)
        batches = queue.Queue(maxsize=self.prefetch_size)
        stop = threading.Event()

        def produce():
            try:
                while not stop.is_set():
                    try:
                        task = requests.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if task is _PREFETCH_END:
                        return
                    item = self.loader.load_batch_for_task(task)
                    if not self._put_unless_stopped(batches, item, stop):
                        return
            except StopIteration:
                self._put_unless_stopped(batches, _PREFETCH_END, stop)
            except BaseException as exc:
                self._put_unless_stopped(
                    batches,
                    _PrefetchException(exc, traceback.format_exc()),
                    stop,
                )

        producer = threading.Thread(
            target=produce,
            name="etpr1-cpu-batch-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            for _ in range(self.prefetch_size):
                task = self.loader.reserve_next_task()
                if not self._put_unless_stopped(requests, task, stop):
                    return

            while True:
                item = batches.get()
                if item is _PREFETCH_END:
                    return
                if isinstance(item, _PrefetchException):
                    raise RuntimeError(
                        "CPU batch prefetch failed:\n"
                        f"{item.formatted_traceback}"
                    ) from item.exception

                task = self.loader.reserve_next_task()
                if not self._put_unless_stopped(requests, task, stop):
                    return
                yield move_to_cuda(item, self.device)
        finally:
            stop.set()
            try:
                requests.put_nowait(_PREFETCH_END)
            except queue.Full:
                pass
            producer.join(timeout=1)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return self.loader.__getattribute__(name)


def build_dataloader(task, dataset, collate_fn, is_train: bool, opts):
    batch_size = opts.train_batch_size if is_train else opts.val_batch_size

    if opts.local_rank == -1:
        if is_train:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size

    else:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=is_train
        )
        pre_epoch = sampler.set_epoch 

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=opts.n_workers,
        pin_memory=opts.pin_mem,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return loader, pre_epoch
