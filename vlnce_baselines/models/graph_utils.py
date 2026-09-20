from collections import defaultdict
from dataclasses import dataclass, replace
import numpy as np
from copy import deepcopy
import networkx as nx
import torch
import matplotlib.pyplot as plt
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff

MAX_DIST = 30
MAX_STEP = 10
# NOISE = 0.5

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    # heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    # Floating-point sqrt/division can put an axis-aligned direction one
    # ULP outside [-1, 1]. Bound only out-of-domain ratios: keep the existing
    # arithmetic and dtype untouched for all previously finite directions.
    heading_ratio = -dx / xz_dist
    if heading_ratio > 1: heading_ratio = 1.0
    elif heading_ratio < -1: heading_ratio = -1.0
    heading = np.arcsin(heading_ratio)  # [-pi/2, pi/2]
    # if b[1] < a[1]:
    #     heading = np.pi - heading
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading

    elevation_ratio = dz / xyz_dist
    if elevation_ratio > 1: elevation_ratio = 1.0
    elif elevation_ratio < -1: elevation_ratio = -1.0
    elevation = np.arcsin(elevation_ratio)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def heading_from_quaternion(quat: np.array):
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

def estimate_cand_pos(pos, ori, ang, dis):
    cand_num = len(ang)
    cand_pos = np.zeros([cand_num, 3])

    ang = np.array(ang)
    dis = np.array(dis)
    ang = (heading_from_quaternion(ori) + ang) % (2 * np.pi)
    cand_pos[:, 0] = pos[0] - dis * np.sin(ang)    # x
    cand_pos[:, 1] = pos[1]                        # y
    cand_pos[:, 2] = pos[2] - dis * np.cos(ang)    # z
    return cand_pos


@dataclass(frozen=True)
class CandidatePreview:
    candidate_vp: str
    target_kind: str
    target_vp: str
    position: object
    front_vp: str
    post_update_ghost_mean: object = None


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y and x !=k and y != k:
                    t_dis = self._dis[x][y] + self._dis[y][k]
                    if t_dis < self._dis[x][k]:
                        self._dis[x][k] = t_dis
                        self._dis[k][x] = t_dis
                        self._point[x][k] = y
                        self._point[k][x] = y

        for x in self._dis:
            for y in self._dis:
                if x != y:
                    t_dis = self._dis[x][k] + self._dis[k][y]
                    if t_dis < self._dis[x][y]:
                        self._dis[x][y] = t_dis
                        self._dis[y][x] = t_dis
                        self._point[x][y] = k
                        self._point[y][x] = k

        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug,
                 ghost_concat_memory_mode="current_step_only"):
        if ghost_concat_memory_mode not in ("current_step_only", "persistent_node_state"):
            raise ValueError("Unknown ghost_concat_memory_mode: %s" % ghost_concat_memory_mode)
        self.ghost_concat_memory_mode = ghost_concat_memory_mode
        self.ghost_concat_state_vps = set()

        self.graph_nx = nx.Graph()

        self.node_pos = {}          # viewpoint to position (x, y, z)
        self.node_embeds = {}       # viewpoint to pano feature
        self.node_stepId = {}

        self.ghost_cnt = 0          # id to create ghost 
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_embeds = {}      # viewpoint to single_view feature
        self.ghost_fronts = {}      # viewpoint to front_vp id
        self.ghost_real_pos = {}    # for training
        self.ghost_goal_dists = {}  # cached geodesic distance for real pos
        self.ghost_persistent_q0 = {}
        self.ghost_candidate_q0 = {}
        self.raenwm_source_contexts = {}
        self.has_real_pos = has_real_pos
        self.merge_ghost = merge_ghost
        self.ghost_aug = ghost_aug  # 0 ~ 1, noise level
        self.loc_noise = loc_noise

        self.shortest_path = None
        self.shortest_dist = None
        
        self.node_stop_scores = {}  # viewpoint to stop_score

    def _localize(self, qpos, kpos_dict, ignore_height=False):
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():
            if ignore_height:
                dis = ((qpos[[0,2]] - kpos[[0,2]])**2).sum()**0.5
            else:
                dis = ((qpos - kpos)**2).sum()**0.5
            if dis < min_dis:
                min_dis = dis
                min_vp = kvp
        min_vp = None if min_dis > self.loc_noise else min_vp
        return min_vp
    
    def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        # assume no repeated node
        # since action is restricted to ghosts
        cur_vp = str(len(self.node_pos)) 
        cand_vp = [f'{cur_vp}_{str(i)}' for i in range(len(cand_ang))]
        cand_pos = [p for p in estimate_cand_pos(cur_pos, cur_ori, cand_ang, cand_dis)]
        return cur_vp, cand_vp, cand_pos

    def preview_candidate_mapping(self, cur_vp, cur_pos, cand_vp, cand_pos):
        if len(cand_vp) != len(cand_pos):
            raise ValueError("candidate viewpoint and position counts must match")
        node_pos = dict(self.node_pos)
        node_pos[cur_vp] = cur_pos
        previews = []
        next_ghost_cnt = int(self.ghost_cnt)
        reserved_ghost_pos = dict(self.ghost_mean_pos)
        reserved_ghost_counts = {
            gvp: len(self.ghost_pos.get(gvp, []))
            for gvp in reserved_ghost_pos
        }
        for cvp, cpos in zip(cand_vp, cand_pos):
            localized_nvp = self._localize(cpos, node_pos)
            if localized_nvp is not None:
                previews.append(CandidatePreview(
                    cvp, "node", localized_nvp, cpos, cur_vp
                ))
                continue

            localized_gvp = None
            if self.merge_ghost:
                localized_gvp = self._localize(cpos, reserved_ghost_pos)
            if localized_gvp is None:
                gvp = f"g{next_ghost_cnt}"
                next_ghost_cnt += 1
                reserved_ghost_pos[gvp] = cpos
                reserved_ghost_counts[gvp] = 1
                previews.append(CandidatePreview(
                    cvp, "new_ghost", gvp, cpos, cur_vp
                ))
            else:
                count = max(
                    1, int(reserved_ghost_counts.get(localized_gvp, 1))
                )
                reserved_ghost_pos[localized_gvp] = (
                    reserved_ghost_pos[localized_gvp] * count + cpos
                ) / (count + 1)
                reserved_ghost_counts[localized_gvp] = count + 1
                kind = (
                    "existing_ghost"
                    if localized_gvp in self.ghost_pos
                    else "new_ghost"
                )
                previews.append(CandidatePreview(
                    cvp, kind, localized_gvp, cpos, cur_vp
                ))
        final_means = {
            ghost_vp: np.asarray(position, dtype=np.float32).copy()
            for ghost_vp, position in reserved_ghost_pos.items()
        }
        return [
            replace(
                preview,
                post_update_ghost_mean=(
                    final_means[str(preview.target_vp)].copy()
                    if preview.target_kind in ("new_ghost", "existing_ghost")
                    else None
                ),
            )
            for preview in previews
        ]

    def write_ghost_concat_state(self, vp, fused_state, *, validated=False):
        """Replace the live state without treating a prediction as an observation.

        The accumulator stores state * real observation count, so the next real
        observation merges as (count * state + observation) / (count + 1).
        Keep autograd history throughout the rollout. ``validated`` is reserved
        for the batched fusion caller, which checks finiteness before writing.
        """
        if self.ghost_concat_memory_mode != "persistent_node_state":
            raise ValueError("Ghost state writeback requires persistent_node_state")
        if vp not in self.ghost_embeds:
            raise KeyError("Unknown ghost vp: %s" % vp)
        accumulator, count = self.ghost_embeds[vp]
        if fused_state.shape != accumulator.shape or count <= 0:
            raise ValueError("Ghost state shape/count mismatch")
        if fused_state.device != accumulator.device or fused_state.dtype != accumulator.dtype:
            raise ValueError("Ghost state device/dtype mismatch")
        replacement = fused_state * count
        if not validated and not bool(torch.isfinite(replacement).all()):
            return False
        self.ghost_embeds[vp] = [replacement, count]
        self.ghost_concat_state_vps.add(vp)
        return True

    def delete_ghost(self, vp):
        self.ghost_concat_state_vps.discard(vp)
        self.ghost_pos.pop(vp)
        self.ghost_mean_pos.pop(vp)
        self.ghost_embeds.pop(vp)
        self.ghost_fronts.pop(vp)
        self.ghost_candidate_q0.pop(vp, None)
        removed_q0 = self.ghost_persistent_q0.pop(vp, None)
        if removed_q0:
            live_context_keys = {
                self._raenwm_source_context_key(
                    record.source_front_vp, record.source_high_level_step
                )
                for records in self.ghost_persistent_q0.values()
                for record in records
            }
            for record in removed_q0:
                key = self._raenwm_source_context_key(
                    record.source_front_vp, record.source_high_level_step
                )
                if key not in live_context_keys:
                    self.raenwm_source_contexts.pop(key, None)
        if self.has_real_pos:
            self.ghost_real_pos.pop(vp)
            self.ghost_goal_dists.pop(vp)

    def record_persistent_q0_candidates(
        self,
        candidate_to_ghost,
        candidate_estimated_positions,
        candidate_real_positions,
        candidate_q0_records,
        candidate_view_indices,
        candidate_forward_distances,
        *,
        source_front_vp,
        source_high_level_step,
        skipped_candidates=None,
    ):
        """Persist trajectory-valid q0 records for the mapped live ghosts."""

        from vlnce_baselines.nwm.active_lookahead.persistent_q0 import (
            append_persistent_q0_records,
            build_persistent_q0_records,
        )

        records = build_persistent_q0_records(
            candidate_to_ghost,
            candidate_estimated_positions,
            candidate_real_positions,
            candidate_q0_records,
            candidate_view_indices,
            candidate_forward_distances,
            source_front_vp=source_front_vp,
            source_high_level_step=source_high_level_step,
            skipped_candidates=skipped_candidates,
        )
        append_persistent_q0_records(self.ghost_persistent_q0, records)
        return records

    def select_persistent_q0(self, ghost_vp):
        """Return the canonical legal q0 using the stable source ordering."""

        if ghost_vp not in self.ghost_mean_pos:
            raise KeyError(f"Unknown ghost vp: {ghost_vp}")
        from vlnce_baselines.nwm.active_lookahead.persistent_q0 import (
            select_ghost_canonical_q0,
        )

        return select_ghost_canonical_q0(
            self.ghost_persistent_q0,
            ghost_vp,
            self.ghost_mean_pos[ghost_vp],
        )

    def replace_candidate_q0_cache(self, observed_ghosts, records):
        """Invalidate observed ghosts, then publish successful latest predictions."""

        from vlnce_baselines.nwm.active_lookahead.types import CandidateQ0

        for ghost_vp in observed_ghosts:
            self.ghost_candidate_q0.pop(str(ghost_vp), None)
        for record in records:
            if not isinstance(record, CandidateQ0):
                raise TypeError("candidate q0 cache must contain CandidateQ0 values")
            ghost_vp = str(record.ghost_vp)
            if ghost_vp not in self.ghost_mean_pos:
                raise KeyError(f"Cannot cache q0 for unknown ghost: {ghost_vp}")
            if not np.allclose(
                np.asarray(record.target_position, dtype=np.float32),
                np.asarray(self.ghost_mean_pos[ghost_vp], dtype=np.float32),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "candidate q0 target differs from committed ghost_mean_pos: "
                    f"{ghost_vp}"
                )
            self.ghost_candidate_q0[ghost_vp] = record
        return tuple(records)

    def select_candidate_q0(self, ghost_vp):
        """Return only the latest reusable prediction; never recompute or fall back."""

        ghost_vp = str(ghost_vp)
        if ghost_vp not in self.ghost_mean_pos:
            raise KeyError(f"Unknown ghost vp: {ghost_vp}")
        record = self.ghost_candidate_q0.get(ghost_vp)
        if record is None:
            return None
        if not np.allclose(
            np.asarray(record.target_position, dtype=np.float32),
            np.asarray(self.ghost_mean_pos[ghost_vp], dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        ):
            self.ghost_candidate_q0.pop(ghost_vp, None)
            return None
        return record

    @staticmethod
    def _raenwm_source_context_key(source_front_vp, source_high_level_step):
        return (str(source_front_vp), int(source_high_level_step))

    def record_raenwm_source_context(self, snapshot):
        key = self._raenwm_source_context_key(
            snapshot.source_front_vp,
            snapshot.source_high_level_step,
        )
        existing = self.raenwm_source_contexts.get(key)
        if existing is not None:
            if existing is not snapshot:
                raise ValueError(f"duplicate RAE-NWM source context for {key}")
            return existing
        self.raenwm_source_contexts[key] = snapshot
        return snapshot

    def get_raenwm_source_context(self, persistent_q0):
        if persistent_q0 is None:
            return None
        return self.raenwm_source_contexts.get(
            self._raenwm_source_context_key(
                persistent_q0.source_front_vp,
                persistent_q0.source_high_level_step,
            )
        )

    def update_graph(self, prev_vp, step_id,
                           cur_vp, cur_pos, cur_embeds,
                           cand_vp, cand_pos, cand_embeds, 
                           cand_real_pos, cand_goal_dists=None,
                           candidate_preview=None):
        if cand_goal_dists is None:
            cand_goal_dists = [None] * len(cand_vp)
        if len(cand_goal_dists) != len(cand_vp):
            raise ValueError(
                "candidate goal-distance count must match candidate count"
            )
        # 1. connect prev_vp
        self.graph_nx.add_node(cur_vp)
        if prev_vp is not None:
            prev_pos = self.node_pos[prev_vp]
            dis = calc_position_distance(prev_pos, cur_pos)
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis)

        # 2. update node & ghost info
        self.node_pos[cur_vp] = cur_pos
        self.node_embeds[cur_vp] = cur_embeds
        self.node_stepId[cur_vp] = step_id
        candidate_to_ghost = []
        if candidate_preview is None:
            candidate_preview = self.preview_candidate_mapping(
                cur_vp, cur_pos, cand_vp, cand_pos
            )
        if len(candidate_preview) != len(cand_vp):
            raise ValueError(
                "candidate_preview length must match cand_vp length: "
                f"{len(candidate_preview)} vs {len(cand_vp)}"
            )
        for i, (cvp, cpos, cembeds, preview) in enumerate(zip(
            cand_vp, cand_pos, cand_embeds, candidate_preview
        )):
            if preview.candidate_vp != cvp:
                raise ValueError(
                    "candidate_preview order must match cand_vp: "
                    f"{preview.candidate_vp} vs {cvp}"
                )
            if preview.target_kind == "node":
                localized_nvp = preview.target_vp
            # cand overlap with node, connect cur_vp with localized_nvp
                dis = calc_position_distance(cur_pos, self.node_pos[localized_nvp])
                self.graph_nx.add_edge(cur_vp, localized_nvp, weight=dis)
                candidate_to_ghost.append((cvp, None))
            # cand not overlap with node, create/update ghost
            elif preview.target_kind in ("new_ghost", "existing_ghost"):
                gvp = preview.target_vp
                if preview.target_kind == "new_ghost" and gvp not in self.ghost_pos:
                    self.ghost_pos[gvp] = [cpos]
                    self.ghost_mean_pos[gvp] = cpos
                    self.ghost_embeds[gvp] = [cembeds, 1]
                    self.ghost_fronts[gvp] = [cur_vp]
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp] = [cand_real_pos[i]]
                        self.ghost_goal_dists[gvp] = [cand_goal_dists[i]]
                    if gvp.startswith("g"):
                        self.ghost_cnt = max(
                            self.ghost_cnt, int(gvp[1:]) + 1
                        )
                else:
                    if gvp not in self.ghost_pos:
                        raise KeyError(f"Unknown preview ghost vp: {gvp}")
                    self.ghost_pos[gvp].append(cpos)
                    self.ghost_mean_pos[gvp] = np.mean(
                        self.ghost_pos[gvp], axis=0
                    )
                    self.ghost_embeds[gvp][0] = (
                        self.ghost_embeds[gvp][0] + cembeds
                    )
                    self.ghost_embeds[gvp][1] += 1
                    self.ghost_fronts[gvp].append(cur_vp)
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp].append(cand_real_pos[i])
                        self.ghost_goal_dists[gvp].append(cand_goal_dists[i])
                candidate_to_ghost.append((cvp, gvp))
            else:
                raise ValueError(
                    "Unsupported candidate preview target_kind: "
                    f"{preview.target_kind}"
                )

        expected_means = {}
        for preview in candidate_preview:
            if preview.target_kind in ("new_ghost", "existing_ghost"):
                expected_means[str(preview.target_vp)] = np.asarray(
                    preview.post_update_ghost_mean, dtype=np.float32
                )
        for ghost_vp, expected_mean in expected_means.items():
            if not np.allclose(
                np.asarray(self.ghost_mean_pos[ghost_vp], dtype=np.float32),
                expected_mean,
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "candidate preview post-update mean differs from graph update: "
                    f"{ghost_vp}"
                )
        
        self.ghost_aug_pos = deepcopy(self.ghost_mean_pos)
        if self.ghost_aug != 0:
            for gvp, gpos in self.ghost_aug_pos.items():
                gpos_noise = np.random.normal(loc=(0,0,0), scale=(self.ghost_aug,0,self.ghost_aug), size=(3,))
                gpos_noise[gpos_noise < -self.ghost_aug] = -self.ghost_aug
                gpos_noise[gpos_noise >  self.ghost_aug] =  self.ghost_aug
                self.ghost_aug_pos[gvp] = gpos + gpos_noise

        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))
        return candidate_to_ghost

    def front_to_ghost_dist(self, ghost_vp):
        # assume the nearest front
        min_dis = 10000
        min_front = None
        for front_vp in self.ghost_fronts[ghost_vp]:
            dis = calc_position_distance(
                self.node_pos[front_vp], self.ghost_aug_pos[ghost_vp]
            )
            if dis < min_dis:
                min_dis = dis
                min_front = front_vp
        return min_dis, min_front

    def get_node_embeds(self, vp):
        if not vp.startswith('g'):
            return self.node_embeds[vp]
        else:
            return self.ghost_embeds[vp][0] / self.ghost_embeds[vp][1]

    def get_pos_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        rel_angles, rel_dists = [], []
        for vp in gmap_vp_ids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            # for ghost
            elif vp.startswith('g'):
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.ghost_aug_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                front_dis, front_vp = self.front_to_ghost_dist(vp)
                shortest_dist = self.shortest_dist[cur_vp][front_vp] + front_dis
                shortest_step = len(self.shortest_path[cur_vp][front_vp]) + 1
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
            # for node
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation, to_clock=True,
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
