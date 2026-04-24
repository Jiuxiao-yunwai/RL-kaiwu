#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import pickle
import time
from collections import deque

import numpy as np

from ..feature.definition import ActData


VIEW_SIZE = 25
MAP_SIZE = 128
UNKNOWN = -1

# Direction ids follow the environment convention:
# 0 East, 1 NorthEast, 2 North, 3 NorthWest,
# 4 West, 5 SouthWest, 6 South, 7 SouthEast.
# After rasterization, grid.x maps to the original z axis and grid.z maps
# to the original x axis, so the offsets are rotated relative to Cartesian x/y.
DIRECTION_DELTAS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


def _in_bounds(x, z, size=MAP_SIZE):
    return 0 <= x < size and 0 <= z < size


def _to_local(hero_pos, target_pos, view_size=VIEW_SIZE):
    dx = target_pos[0] - hero_pos.x
    dz = target_pos[1] - hero_pos.z
    if abs(dx) > view_size or abs(dz) > view_size:
        return None
    return view_size + dx, view_size + dz


def _direction_from_delta(dx, dz):
    step = (
        0 if dx == 0 else (1 if dx > 0 else -1),
        0 if dz == 0 else (1 if dz > 0 else -1),
    )
    for idx, delta in enumerate(DIRECTION_DELTAS):
        if delta == step:
            return idx
    return None


class GlobalMap:
    def __init__(self, size=MAP_SIZE, view_size=VIEW_SIZE):
        self.size = size
        self.view_size = view_size
        self.passable = np.full((size, size), UNKNOWN, dtype=np.int8)
        self.observed = np.zeros((size, size), dtype=np.uint8)
        self.treasure_pos = set()
        self.buff_pos = None
        self.end_pos = None

    @property
    def explored_ratio(self):
        return float(self.observed.mean())

    def update_local(self, hero_pos, obstacle_map):
        local_map = np.asarray(obstacle_map, dtype=np.int8).reshape((self.view_size * 2 + 1, self.view_size * 2 + 1))
        for local_x in range(local_map.shape[0]):
            for local_z in range(local_map.shape[1]):
                global_x = hero_pos.x + (local_x - self.view_size)
                global_z = hero_pos.z + (local_z - self.view_size)
                if not _in_bounds(global_x, global_z, self.size):
                    continue
                self.passable[global_x, global_z] = 1 if local_map[local_x, local_z] > 0 else 0
                self.observed[global_x, global_z] = 1

    def update_organs(self, organs, get_grid_pos):
        for organ in organs:
            organ_grid_pos = get_grid_pos(organ.pos.x, organ.pos.z)
            pos = (int(organ_grid_pos.x), int(organ_grid_pos.z))
            if organ.sub_type == 1:
                if organ.status == 1:
                    self.treasure_pos.add(pos)
                else:
                    self.treasure_pos.discard(pos)
            elif organ.sub_type == 2:
                self.buff_pos = pos if organ.status == 1 else None

    def update_end(self, end_pos):
        self.end_pos = tuple(end_pos) if end_pos is not None else None

    def save(self, path, id="1"):
        os.makedirs(path, exist_ok=True)
        state = {
            "passable": self.passable,
            "observed": self.observed,
            "treasure_pos": sorted(self.treasure_pos),
            "buff_pos": self.buff_pos,
            "end_pos": self.end_pos,
        }
        file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
        return file_path

    def load(self, path, id="1"):
        file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        if not os.path.exists(file_path):
            return False

        with open(file_path, "rb") as f:
            state = pickle.load(f)

        self.passable = state.get("passable", self.passable)
        self.observed = state.get("observed", self.observed)
        self.treasure_pos = set(tuple(item) for item in state.get("treasure_pos", []))
        self.buff_pos = state.get("buff_pos", self.buff_pos)
        self.end_pos = state.get("end_pos", self.end_pos)
        return True


class Algorithm:
    def __init__(self, logger=None, monitor=None):
        self.logger = logger
        self.monitor = monitor
        self.gmap = GlobalMap()
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        now = time.time()
        if self.monitor and now - self.last_report_monitor_time >= 60:
            self.monitor.put_data(
                {
                    os.getpid(): {
                        "reward": float(sum(frame.rew for frame in list_sample_data)) if list_sample_data else 0.0,
                        "diy_1": float(self.gmap.explored_ratio),
                    }
                }
            )
            self.last_report_monitor_time = now

    def plan_action(self, hero_pos, remain_info, obstacle_map, treasure_map, end_map, legal_act):
        if hero_pos is None:
            return ActData(move_dir=0, use_talent=0)

        local_map = np.asarray(obstacle_map, dtype=np.int8).reshape((VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1))
        self.gmap.update_local(hero_pos, local_map)

        if not legal_act or not bool(legal_act[0]):
            return ActData(move_dir=0, use_talent=0)

        target = self._choose_target(hero_pos, remain_info)
        direction = None
        if target is not None:
            direction = self._plan_towards_target(hero_pos, local_map, target)

        if direction is None:
            direction = self._plan_explore(hero_pos, local_map)

        if direction is None:
            direction = self._fallback_direction(hero_pos, local_map, remain_info)

        return ActData(move_dir=int(direction), use_talent=0)

    def _choose_target(self, hero_pos, remain_info):
        self.gmap.treasure_pos.discard((hero_pos.x, hero_pos.z))

        treasure_total = int(remain_info.get("treasure_count", 0)) if remain_info else 0
        treasure_collected = int(remain_info.get("treasure_collected_count", 0)) if remain_info else 0

        if treasure_collected < treasure_total and self.gmap.treasure_pos:
            return min(self.gmap.treasure_pos, key=lambda pos: self._target_priority(hero_pos, pos))

        if self.gmap.end_pos is not None:
            return self.gmap.end_pos

        return None

    def _target_priority(self, hero_pos, target_pos):
        local_target = _to_local(hero_pos, target_pos)
        if local_target is not None:
            return abs(local_target[0] - VIEW_SIZE) + abs(local_target[1] - VIEW_SIZE)
        return abs(target_pos[0] - hero_pos.x) + abs(target_pos[1] - hero_pos.z)

    def _plan_towards_target(self, hero_pos, local_map, target_pos):
        local_goal = _to_local(hero_pos, target_pos)
        if local_goal is None:
            return _direction_from_delta(target_pos[0] - hero_pos.x, target_pos[1] - hero_pos.z)

        direction = self._bfs_first_direction(local_map, {local_goal})
        if direction is not None:
            return direction

        return _direction_from_delta(target_pos[0] - hero_pos.x, target_pos[1] - hero_pos.z)

    def _plan_explore(self, hero_pos, local_map):
        frontier_goals = set()
        center = (VIEW_SIZE, VIEW_SIZE)

        for local_x in range(local_map.shape[0]):
            for local_z in range(local_map.shape[1]):
                if (local_x, local_z) == center or local_map[local_x, local_z] <= 0:
                    continue

                global_x = hero_pos.x + (local_x - VIEW_SIZE)
                global_z = hero_pos.z + (local_z - VIEW_SIZE)
                if not _in_bounds(global_x, global_z):
                    continue

                for dx, dz in DIRECTION_DELTAS:
                    nx = global_x + dx
                    nz = global_z + dz
                    if _in_bounds(nx, nz) and self.gmap.observed[nx, nz] == 0:
                        frontier_goals.add((local_x, local_z))
                        break

        if not frontier_goals:
            return None

        return self._bfs_first_direction(local_map, frontier_goals)

    def _fallback_direction(self, hero_pos, local_map, remain_info):
        recent_position_map = remain_info.get("recent_position_map", {}) if remain_info else {}
        best_direction = None
        best_score = None

        for direction, (dx, dz) in enumerate(DIRECTION_DELTAS):
            local_x = VIEW_SIZE + dx
            local_z = VIEW_SIZE + dz
            if not (0 <= local_x < local_map.shape[0] and 0 <= local_z < local_map.shape[1]):
                continue
            if local_map[local_x, local_z] <= 0:
                continue

            next_pos = (hero_pos.x + dx, hero_pos.z + dz)
            visit_count = int(recent_position_map.get(next_pos, 0))
            observed_bonus = int(self.gmap.observed[next_pos]) if _in_bounds(*next_pos) else 1
            score = (visit_count, observed_bonus)

            if best_score is None or score < best_score:
                best_score = score
                best_direction = direction

        return best_direction if best_direction is not None else 0

    def _bfs_first_direction(self, local_map, goals):
        start = (VIEW_SIZE, VIEW_SIZE)
        if start in goals:
            return None

        queue = deque([(start, None)])
        visited = {start}

        while queue:
            (x, z), first_direction = queue.popleft()
            for direction, (dx, dz) in enumerate(DIRECTION_DELTAS):
                nx = x + dx
                nz = z + dz
                if not (0 <= nx < local_map.shape[0] and 0 <= nz < local_map.shape[1]):
                    continue
                if local_map[nx, nz] <= 0 or (nx, nz) in visited:
                    continue

                next_first = direction if first_direction is None else first_direction
                if (nx, nz) in goals:
                    return next_first

                visited.add((nx, nz))
                queue.append(((nx, nz), next_first))

        return None
