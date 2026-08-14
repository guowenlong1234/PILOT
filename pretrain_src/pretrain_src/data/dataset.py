'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math
import mmap

from array import array
from bisect import bisect_right
from collections import OrderedDict

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax

MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20


class IndexedJsonlSequence:
    """Random-access JSONL sequence without expanding every row in memory.

    Only compact byte offsets are kept in Python memory.  Each process opens
    read-only mmap handles lazily, so DataLoader ``spawn`` workers neither copy
    millions of decoded dictionaries nor inherit live file handles.
    """

    def __init__(self, paths, selected_indices=None):
        self.paths = tuple(os.path.abspath(os.fspath(path)) for path in paths)
        self._offsets = []
        self._file_sizes = []
        self._cumulative_sizes = [0]

        for path in self.paths:
            offsets = array('Q')
            next_offset = 0
            with open(path, 'rb') as handle:
                for line in handle:
                    offsets.append(next_offset)
                    next_offset += len(line)
            self._offsets.append(offsets)
            self._file_sizes.append(next_offset)
            self._cumulative_sizes.append(
                self._cumulative_sizes[-1] + len(offsets)
            )

        self._selected_indices = None
        if selected_indices is not None:
            self.select(selected_indices)

        self._handles = {}
        self._mmaps = {}

    def __len__(self):
        if self._selected_indices is not None:
            return len(self._selected_indices)
        return self._cumulative_sizes[-1]

    def _normalize_index(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('JSONL index out of range')
        if self._selected_indices is not None:
            index = self._selected_indices[index]
        return index

    def _get_mmap(self, file_index):
        mapped = self._mmaps.get(file_index)
        if mapped is None:
            handle = open(self.paths[file_index], 'rb')
            try:
                mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except BaseException:
                handle.close()
                raise
            self._handles[file_index] = handle
            self._mmaps[file_index] = mapped
        return mapped

    def select(self, selected_indices):
        selected_values = [int(index) for index in selected_indices]
        total = self._cumulative_sizes[-1]
        if any(index < 0 or index >= total for index in selected_values):
            raise IndexError('selected JSONL index is out of range')
        self._selected_indices = array('Q', selected_values)
        return self

    def __getitem__(self, index):
        index = self._normalize_index(index)
        file_index = bisect_right(self._cumulative_sizes, index) - 1
        local_index = index - self._cumulative_sizes[file_index]
        offsets = self._offsets[file_index]
        start = offsets[local_index]
        if local_index + 1 < len(offsets):
            end = offsets[local_index + 1]
        else:
            end = self._file_sizes[file_index]
        return json.loads(self._get_mmap(file_index)[start:end])

    @property
    def index_bytes(self):
        selected_bytes = (
            0
            if self._selected_indices is None
            else self._selected_indices.itemsize * len(self._selected_indices)
        )
        return selected_bytes + sum(
            offsets.itemsize * len(offsets) for offsets in self._offsets
        )

    def close(self):
        for mapped in self._mmaps.values():
            mapped.close()
        for handle in self._handles.values():
            handle.close()
        self._mmaps = {}
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = {}
        state['_mmaps'] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _feature_payload_nbytes(value):
    """Count array payload bytes stored by the feature cache."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_feature_payload_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_feature_payload_nbytes(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 0


class ByteLRUCache:
    """Least-recently-used cache bounded by NumPy payload bytes."""

    def __init__(self, max_bytes=None):
        if max_bytes is not None and int(max_bytes) < 0:
            raise ValueError('feature cache max_bytes cannot be negative')
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.skipped = 0
        self._items = OrderedDict()

    def get(self, key):
        try:
            value, size = self._items.pop(key)
        except KeyError:
            self.misses += 1
            return None
        self._items[key] = (value, size)
        self.hits += 1
        return value

    def put(self, key, value):
        size = _feature_payload_nbytes(value)
        previous = self._items.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]

        if self.max_bytes is not None and size > self.max_bytes:
            self.skipped += 1
            return False

        while (
            self.max_bytes is not None
            and self.current_bytes + size > self.max_bytes
            and self._items
        ):
            _, (_, evicted_size) = self._items.popitem(last=False)
            self.current_bytes -= evicted_size
            self.evictions += 1

        self._items[key] = (value, size)
        self.current_bytes += size
        return True

    def __len__(self):
        return len(self._items)

    def info(self):
        return {
            'entries': len(self),
            'current_bytes': self.current_bytes,
            'max_bytes': self.max_bytes,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'skipped': self.skipped,
        }

RAE_DINO_HDF5_METADATA = {
    'feature_extractor': 'rae_dinov2_with_registers_base_raw_cls',
    'feature_dim': 768,
    'dtype': 'float32',
    'num_views': 36,
    'image_size': 224,
    'vfov': 60,
    'cls_normalization': 'none',
    'rae_stat_applied_to_cls': False,
}


def _validation_sample_indices(num_items, sample_num, seed=None):
    if not sample_num:
        return None
    rng = np.random if seed is None else np.random.RandomState(int(seed))
    return rng.permutation(int(num_items))[: int(sample_num)]


def _normalize_hdf5_attr(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_value_matches(got, expected):
    return type(got) is type(expected) and got == expected

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None, val_sample_seed=None,
        raw_image_feat_size=None, rgb_encoder_type='clip',
        feature_cache_size_mb=None, lazy_load_annotations=True,
    ):
        self.img_ft_file = img_ft_file
        self.dep_ft_file = dep_ft_file
        self.obj_ft_file = obj_ft_file

        self.rgb_encoder_type = str(rgb_encoder_type).lower()
        self.image_feat_size = int(image_feat_size)
        self.raw_image_feat_size = (
            self.image_feat_size
            if raw_image_feat_size is None else int(raw_image_feat_size)
        )
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.depth_feat_size = depth_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        if self.rgb_encoder_type == 'rae_dinov2':
            self._validate_rae_dino_metadata()

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            cache_max_bytes = (
                None
                if feature_cache_size_mb is None
                else int(float(feature_cache_size_mb) * 1024 * 1024)
            )
            self._feature_store = ByteLRUCache(cache_max_bytes)

        self.scanvp_cands = json.load(open(scanvp_cands_file))

        
        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir) 
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)] 
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles] 

        if lazy_load_annotations:
            indexed_data = IndexedJsonlSequence(anno_files)
            if val_sample_num:
                # cannot evaluate all the samples as it takes too much time
                sel_idxs = _validation_sample_indices(
                    len(indexed_data), val_sample_num, seed=val_sample_seed
                )
                indexed_data.select(sel_idxs)
            self.data = indexed_data
        else:
            self.data = []
            for anno_file in anno_files:
                with jsonlines.open(anno_file, 'r') as f:
                    for item in f:
                        self.data.append(item)

            if val_sample_num:
                sel_idxs = _validation_sample_indices(
                    len(self.data), val_sample_num, seed=val_sample_seed
                )
                self.data = [self.data[sidx] for sidx in sel_idxs]

    def _validate_rae_dino_metadata(self):
        with h5py.File(self.img_ft_file, 'r') as handle:
            for key, expected in RAE_DINO_HDF5_METADATA.items():
                got = _normalize_hdf5_attr(handle.attrs.get(key))
                if not _metadata_value_matches(got, expected):
                    raise ValueError(
                        f'DINO HDF5 metadata mismatch for {key}: '
                        f'expected {expected!r} ({type(expected).__name__}), '
                        f'got {got!r} ({type(got).__name__})'
                    )

    def _validate_view_features(self, key, view_fts):
        shape = tuple(view_fts.shape)
        expected = (
            'expected a two-dimensional array with 36 views and at least '
            f'{self.raw_image_feat_size} feature columns'
        )
        if view_fts.ndim != 2:
            raise ValueError(
                f'RGB feature {key!r} has actual shape {shape}; {expected}'
            )
        if shape[0] != 36:
            raise ValueError(
                f'RGB feature {key!r} has actual shape {shape}; {expected}'
            )
        if shape[1] < self.raw_image_feat_size:
            raise ValueError(
                f'RGB feature {key!r} has actual shape {shape}; {expected}'
            )

    def __len__(self):
        return len(self.data)

    def feature_cache_info(self):
        if not self.in_memory:
            return {
                'entries': 0, 'current_bytes': 0, 'max_bytes': 0,
                'hits': 0, 'misses': 0, 'evictions': 0, 'skipped': 0,
            }
        return self._feature_store.info()

    def close(self):
        data = getattr(self, 'data', None)
        if hasattr(data, 'close'):
            data.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        cached = self._feature_store.get(key) if self.in_memory else None
        if cached is not None:
            view_fts, obj_fts, obj_attrs = cached
        else:
            with h5py.File(self.img_ft_file, 'r') as img_file:
                view_fts = img_file[key][...].astype(np.float32)
            self._validate_view_features(key, view_fts)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as obj_file:
                    if key in obj_file:
                        obj_fts = obj_file[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in obj_file[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
            if self.in_memory:
                self._feature_store.put(key, (view_fts, obj_fts, obj_attrs))

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            obj_label = -100 # ignore 
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.raw_image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.raw_image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0) 
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys()) 
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node: 
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]] / MAX_DIST
        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
       

       

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_file, dep_ft_file, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, depth_feat_size=128, angle_feat_size=4,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        val_sample_num=None, val_sample_seed=None, start_vp_file=None,
        raw_image_feat_size=None, rgb_encoder_type='clip',
        feature_cache_size_mb=None, lazy_load_annotations=True,
    ):
        super().__init__(
            anno_files, img_ft_file, dep_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size, depth_feat_size=depth_feat_size,
            raw_image_feat_size=raw_image_feat_size, rgb_encoder_type=rgb_encoder_type,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node, val_sample_num=val_sample_num,
            val_sample_seed=val_sample_seed,
            feature_cache_size_mb=feature_cache_size_mb,
            lazy_load_annotations=lazy_load_annotations,
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        cached = self._feature_store.get(key) if self.in_memory else None
        if cached is not None:
            view_fts, dep_fts = cached
        else:
            with h5py.File(self.img_ft_file, 'r') as img_file:
                view_fts = img_file[key][...].astype(np.float32)
            self._validate_view_features(key, view_fts)
            with h5py.File(self.dep_ft_file, 'r') as dep_file:
                dep_fts = dep_file[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store.put(key, (view_fts, dep_fts))
        return view_fts, dep_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]: 
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1] 
                end_idx = np.random.randint(len(end_vps)) 
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading) 

        if len(gt_path) > TRAIN_MAX_STEP:
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path)

        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len], # ID
            'task_type_encoding': item['task_type_encoding'],
            
            'traj_view_img_fts': [x[:, :self.raw_image_feat_size] for x in traj_view_img_fts],
            'traj_view_dep_fts': [x[:, :self.depth_feat_size] for x in traj_view_dep_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,
        }

        if return_act_label: 
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.raw_image_feat_size:], dim=1)
        
        return outs

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, dep_fts = self.get_scanvp_feature(scan, vp) 

            view_img_fts, view_dep_fts, view_angles, cand_vpids = [], [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)] 
            used_viewidxs = set()
            for k, v in nav_cands.items(): 
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]]) 
                view_dep_fts.append(dep_fts[v[0]]) 
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]]) 
                cand_vpids.append(k) 
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_dep_fts.extend([dep_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs]) 

            view_img_fts = np.stack(view_img_fts, 0) 
            view_dep_fts = np.stack(view_dep_fts, 0) 
            view_angles = np.stack(view_angles, 0) 
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size) 
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_view_dep_fts.append(view_dep_fts)
            traj_loc_fts.append(view_ang_fts)
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)
            
            last_vp_angles = view_angles 

        return traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles
