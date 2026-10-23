"""Static M-major scheduling for the Wan convolution's one- or two-CTA clusters."""

from dataclasses import dataclass

import cutlass
from cutlass import cute


@dataclass(frozen=True)
class StaticTileSchedule:
    """Compile-time tile geometry; grid Z identifies persistent workers."""

    tiles_m: int
    tiles_n: int
    cluster_m: int

    @property
    def clusters_m(self) -> int:
        """Round up M tiles so a partial final CTA pair still owns a full cluster."""
        return (self.tiles_m + self.cluster_m - 1) // self.cluster_m

    def grid_shape(self, max_clusters: int) -> tuple[int, int, int]:
        """Limit persistent workers without changing the cluster's X extent."""
        return (self.cluster_m, 1, min(self.clusters_m * self.tiles_n, max_clusters))

    @cute.jit
    def get_work(self, index: cutlass.Int32):
        """Decode a cluster index; callers advance it by gridDim.z after each tile."""
        tile_n = index // self.clusters_m
        cluster_m = index % self.clusters_m
        tile_m = cluster_m * self.cluster_m + cute.arch.block_idx()[0]
        return (
            (
                cute.arch.make_warp_uniform(tile_m),
                cute.arch.make_warp_uniform(tile_n),
                cutlass.Int32(0),
            ),
            cutlass.Boolean(cute.arch.make_warp_uniform(index < self.clusters_m * self.tiles_n)),
        )
