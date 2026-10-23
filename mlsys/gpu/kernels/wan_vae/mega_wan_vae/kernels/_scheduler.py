"""M-major scheduling with fixed tile ordering and static or runtime M extent."""

from dataclasses import dataclass

import cutlass
from cutlass import cute


@dataclass(frozen=True)
class TileSchedule:
    """Compile-time tile geometry; grid Z identifies persistent workers.

    Convolutions supply their M extent at runtime. Static schedules remain
    available for standalone callers with a fixed work domain.
    """

    tiles_n: int
    cluster_m: int
    tiles_m: int | None = None  # None explicitly requires a runtime M extent.

    @property
    def clusters_m(self) -> int:
        """Round up M tiles so a partial final CTA pair still owns a full cluster."""
        if self.tiles_m is None:
            raise ValueError("This schedule requires a runtime M extent")
        return (self.tiles_m + self.cluster_m - 1) // self.cluster_m

    def grid_shape(self, max_clusters: int) -> tuple[int, int, int]:
        """Limit persistent workers without changing the cluster's X extent."""
        return (self.cluster_m, 1, min(self.clusters_m * self.tiles_n, max_clusters))

    @cute.jit
    def get_work(self, index: cutlass.Int32, runtime_clusters_m=None, wide_index: cutlass.Constexpr = False):
        """Decode a cluster index; callers advance it by gridDim.z after each tile."""
        if cutlass.const_expr(runtime_clusters_m is None):
            clusters_m = self.clusters_m
            divisor = self.clusters_m
        elif cutlass.const_expr(wide_index):
            clusters_m = runtime_clusters_m
            divisor = runtime_clusters_m
        else:
            clusters_m = runtime_clusters_m.divisor
            divisor = runtime_clusters_m
        tile_n = index // divisor
        cluster_m = index % divisor
        tile_m = cluster_m * self.cluster_m + cute.arch.block_idx()[0]
        return (
            (
                cute.arch.make_warp_uniform(tile_m),
                cute.arch.make_warp_uniform(tile_n),
                cutlass.Int32(0),
            ),
            cutlass.Boolean(cute.arch.make_warp_uniform(index < clusters_m * self.tiles_n)),
        )
