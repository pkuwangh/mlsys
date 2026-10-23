"""Named runtime geometry passed through CuTe host/device boundaries."""

import cutlass
from cutlass import cute
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values


class SpatialDivisors:
    """Host-precomputed divisors for row-major T/H/W coordinate decomposition.

    Narrow variants require the complete row-index space to fit signed Int32.
    Wide variants preserve Int64 division. Tensor addresses always use Int64.
    By default the batch coordinate is widened before downstream arithmetic.
    Callers that keep bounded pixel arithmetic narrow must widen the flattened
    pixel index before multiplying by channels or byte strides.
    """

    def __init__(self, frames, height, width, *, wide=False):
        self.wide = wide
        create = (lambda value: value) if wide else cute.fast_divmod_create_divisor
        self.width = create(width)
        self.plane = create(height * width)
        self.volume = create(frames * height * width)

    def __extract_mlir_values__(self):
        return extract_mlir_values((self.width, self.plane, self.volume))

    def __new_from_mlir_values__(self, values):
        result = object.__new__(SpatialDivisors)
        result.wide = self.wide
        result.width, result.plane, result.volume = new_from_mlir_values((self.width, self.plane, self.volume), values)
        return result

    @cute.jit
    def plane_coordinates(self, row, *, promote_batch: cutlass.Constexpr[bool] = True):
        """Decode flattened N*T,H,W without a redundant temporal division."""
        index = (cutlass.Int64 if self.wide else cutlass.Int32)(row)
        plane = index // self.plane
        remainder = index % self.plane
        return (cutlass.Int64(plane) if promote_batch else plane), remainder // self.width, remainder % self.width

    @cute.jit
    def coordinates(self, row, *, promote_batch: cutlass.Constexpr[bool] = True):
        index = (cutlass.Int64 if self.wide else cutlass.Int32)(row)
        batch = index // self.volume
        remainder = index % self.volume
        frame = remainder // self.plane
        remainder = remainder % self.plane
        height = remainder // self.width
        width = remainder % self.width
        # Promotion here also protects all subsequent flattened pointer products.
        return (cutlass.Int64(batch) if promote_batch else batch), frame, height, width
