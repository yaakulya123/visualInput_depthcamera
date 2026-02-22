#!/usr/bin/env python3
"""
People Cluster Detector - Groups people by 3D proximity using union-find.

Pure computation module, no OpenCV dependency. Takes a list of PersonPositions
(with real-world 3D coordinates from RealSense deprojection) and clusters
people who are within a distance threshold of each other.

Uses union-find (disjoint set) for efficient clustering:
  1. Compute all pairwise 3D Euclidean distances
  2. Union pairs closer than threshold
  3. Cluster ID = min track_id in each group (stable coloring)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PersonPosition:
    """3D position of a tracked person."""
    track_id: int
    pixel_x: float          # Pixel coordinate (for drawing)
    pixel_y: float
    world_x: float = 0.0    # Real-world meters (from rs2_deproject_pixel_to_point)
    world_y: float = 0.0
    world_z: float = 0.0    # Depth in meters
    depth_valid: bool = False


@dataclass
class PairDistance:
    """Distance between two people."""
    id_a: int
    id_b: int
    distance_m: float       # 3D Euclidean distance in meters


@dataclass
class ClusterResult:
    """Result of clustering computation."""
    clusters: Dict[int, List[int]] = field(default_factory=dict)
    # cluster_id (= min track_id in group) -> list of track_ids
    person_to_cluster: Dict[int, int] = field(default_factory=dict)
    # track_id -> cluster_id
    pairwise_distances: List[PairDistance] = field(default_factory=list)
    # All pairs with computed distances

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def person_count(self) -> int:
        return len(self.person_to_cluster)


class _UnionFind:
    """Simple union-find / disjoint set data structure."""

    def __init__(self, elements: List[int]):
        self.parent: Dict[int, int] = {e: e for e in elements}
        self.rank: Dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class PeopleClusterDetector:
    """
    Clusters people by 3D proximity using union-find.

    Persons with depth_valid=False become singleton clusters.
    """

    def __init__(self, threshold_meters: float = 1.0):
        self.threshold_meters = threshold_meters

    def compute(self, positions: List[PersonPosition]) -> ClusterResult:
        if not positions:
            return ClusterResult()

        ids = [p.track_id for p in positions]
        pos_map = {p.track_id: p for p in positions}

        # Compute all pairwise 3D distances
        pairwise: List[PairDistance] = []
        valid_ids = [p.track_id for p in positions if p.depth_valid]

        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                a = pos_map[valid_ids[i]]
                b = pos_map[valid_ids[j]]
                dist = math.sqrt(
                    (a.world_x - b.world_x) ** 2 +
                    (a.world_y - b.world_y) ** 2 +
                    (a.world_z - b.world_z) ** 2
                )
                pairwise.append(PairDistance(a.track_id, b.track_id, dist))

        # Add inf distances for pairs involving invalid-depth persons
        invalid_ids = [p.track_id for p in positions if not p.depth_valid]
        for inv_id in invalid_ids:
            for other_id in ids:
                if other_id != inv_id:
                    pairwise.append(PairDistance(inv_id, other_id, float('inf')))

        # Union-find: merge pairs within threshold
        uf = _UnionFind(ids)
        for pair in pairwise:
            if pair.distance_m <= self.threshold_meters:
                uf.union(pair.id_a, pair.id_b)

        # Build clusters: cluster_id = min track_id in group
        root_to_members: Dict[int, List[int]] = {}
        for tid in ids:
            root = uf.find(tid)
            root_to_members.setdefault(root, []).append(tid)

        clusters: Dict[int, List[int]] = {}
        person_to_cluster: Dict[int, int] = {}
        for members in root_to_members.values():
            cluster_id = min(members)
            clusters[cluster_id] = sorted(members)
            for m in members:
                person_to_cluster[m] = cluster_id

        return ClusterResult(
            clusters=clusters,
            person_to_cluster=person_to_cluster,
            pairwise_distances=pairwise,
        )
