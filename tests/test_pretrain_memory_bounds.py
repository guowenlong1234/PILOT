import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain_src.pretrain_src.data import loader as loader_module
from pretrain_src.pretrain_src.data.loader import MetaLoader
from pretrain_src.pretrain_src.data.dataset import (
    ByteLRUCache,
    IndexedJsonlSequence,
    R2RTextPathData,
)


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_indexed_jsonl_reads_multiple_files_without_decoding_them_eagerly(
    tmp_path,
):
    first = tmp_path / "first.jsonl"
    empty = tmp_path / "empty.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        {"id": 0, "text": "第一条"},
        {"id": 1, "text": "second"},
        {"id": 2, "text": "第三条"},
    ]
    _write_jsonl(first, rows[:2])
    empty.write_bytes(b"")
    _write_jsonl(second, rows[2:])

    data = IndexedJsonlSequence([first, empty, second])

    assert len(data) == 3
    assert data.index_bytes == 3 * 8
    assert [data[index] for index in range(len(data))] == rows
    assert data[-1] == rows[-1]


def test_indexed_jsonl_selection_and_pickle_drop_open_mmap_handles(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [{"id": index} for index in range(5)]
    _write_jsonl(path, rows)
    data = IndexedJsonlSequence([path]).select([4, 1, 3])

    assert data[0] == rows[4]
    assert data._mmaps

    restored = pickle.loads(pickle.dumps(data))

    assert restored._mmaps == {}
    assert restored._handles == {}
    assert [restored[index] for index in range(len(restored))] == [
        rows[4],
        rows[1],
        rows[3],
    ]


def test_indexed_jsonl_is_usable_by_spawn_workers(tmp_path):
    path = tmp_path / "spawn.jsonl"
    _write_jsonl(path, [{"id": index} for index in range(6)])
    data = IndexedJsonlSequence([path])
    # Open the parent mmap first: spawn workers must still receive no live handle.
    assert data[0]["id"] == 0

    loader = DataLoader(
        data,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        prefetch_factor=1,
    )

    ids = torch.cat([batch["id"] for batch in loader]).tolist()
    assert ids == list(range(6))


def test_byte_lru_cache_enforces_payload_limit_and_recency():
    cache = ByteLRUCache(max_bytes=16)
    first = np.zeros(2, dtype=np.float32)
    second = np.ones(2, dtype=np.float32)
    third = np.full(2, 2, dtype=np.float32)

    assert cache.put("first", first)
    assert cache.put("second", second)
    assert cache.current_bytes == 16
    assert cache.get("first") is first

    assert cache.put("third", third)
    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    assert cache.info() == {
        "entries": 2,
        "current_bytes": 16,
        "max_bytes": 16,
        "hits": 3,
        "misses": 1,
        "evictions": 1,
        "skipped": 0,
    }


def test_byte_lru_cache_skips_an_item_larger_than_the_limit():
    cache = ByteLRUCache(max_bytes=4)

    assert not cache.put("too-large", np.zeros(2, dtype=np.float32))
    assert cache.info()["entries"] == 0
    assert cache.info()["current_bytes"] == 0
    assert cache.info()["skipped"] == 1


def test_r2r_rgb_and_depth_are_evicted_as_one_cache_entry(tmp_path):
    image_path = tmp_path / "rgb.hdf5"
    depth_path = tmp_path / "depth.hdf5"
    keys = ("scan_first", "scan_second")
    with h5py.File(image_path, "w") as handle:
        for index, key in enumerate(keys):
            handle.create_dataset(
                key, data=np.full((36, 4), index, dtype=np.float32)
            )
    with h5py.File(depth_path, "w") as handle:
        for index, key in enumerate(keys):
            handle.create_dataset(
                key, data=np.full((36, 2), index, dtype=np.float32)
            )

    pair_bytes = (36 * 4 + 36 * 2) * np.dtype(np.float32).itemsize
    dataset = object.__new__(R2RTextPathData)
    dataset.img_ft_file = str(image_path)
    dataset.dep_ft_file = str(depth_path)
    dataset.raw_image_feat_size = 4
    dataset.image_feat_size = 4
    dataset.in_memory = True
    dataset._feature_store = ByteLRUCache(pair_bytes)
    dataset.data = []

    first_rgb, first_depth = dataset.get_scanvp_feature("scan", "first")
    dataset.get_scanvp_feature("scan", "second")
    info = dataset.feature_cache_info()

    assert first_rgb.shape == (36, 4)
    assert first_depth.shape == (36, 2)
    assert info["entries"] == 1
    assert info["current_bytes"] == pair_bytes
    assert info["evictions"] == 1
    dataset.close()


def test_build_dataloader_keeps_two_training_workers_but_zero_validation_workers(
    monkeypatch,
):
    calls = []

    def capture_dataloader(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(loader_module, "DataLoader", capture_dataloader)
    opts = SimpleNamespace(
        train_batch_size=16,
        val_batch_size=32,
        local_rank=-1,
        n_workers=2,
        val_n_workers=0,
        pin_mem=False,
        dataloader_start_method="spawn",
        prefetch_factor=1,
        persistent_workers=False,
    )
    dataset = list(range(8))

    loader_module.build_dataloader("mlm", dataset, list, True, opts)
    loader_module.build_dataloader("mlm", dataset, list, False, opts)

    training, validation = calls
    assert training["num_workers"] == 2
    assert training["multiprocessing_context"] == "spawn"
    assert training["prefetch_factor"] == 1
    assert training["persistent_workers"] is False
    assert training["pin_memory"] is False
    assert validation["num_workers"] == 0
    assert "multiprocessing_context" not in validation
    assert "prefetch_factor" not in validation
    assert "persistent_workers" not in validation


def test_meta_loader_close_shuts_workers_down_once_and_is_idempotent():
    class FakeIterator:
        def __init__(self):
            self.shutdown_calls = 0

        def _shutdown_workers(self):
            self.shutdown_calls += 1

    iterator = FakeIterator()
    loader = SimpleNamespace(_iterator=iterator)
    meta_loader = object.__new__(MetaLoader)
    meta_loader.name2iter = {"mlm": iterator}
    meta_loader.name2loader = {"mlm": loader}

    meta_loader.close()
    meta_loader.close()

    assert iterator.shutdown_calls == 1
    assert loader._iterator is None
    assert meta_loader.name2iter == {}
