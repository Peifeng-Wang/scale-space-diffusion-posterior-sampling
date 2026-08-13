import torch
import torch.nn.functional as F


class DownsamplingTransfer:
    def __init__(
        self,
        shape,
        factor=2,
        dtype=torch.float32,
        padding="replicate",
        device="cuda",
    ):
        self.shape = shape
        self.factor = factor
        self.dtype = dtype
        self.padding = padding
        self.device = device

    def projection(self, x):
        return F.interpolate(
            x,
            scale_factor=1 / self.factor,
            mode="area",
        )

    def prolongation(self, x):
        return F.interpolate(
            x,
            scale_factor=self.factor,
            mode="bilinear",
            align_corners=False,
        )


class MultiLevelTransfer:
    def __init__(
        self,
        n_level,
        init_shape,
        factor=2,
        dtype=torch.float32,
        padding="circular",
        device="cuda",
    ):
        self.ops = []
        self.n_levels = n_level
        self.factor = factor
        self.init_shape = init_shape
        self.dtype = dtype
        self.padding = padding
        self.device = device

        div_factor = 1
        for _ in range(n_level - 1):
            shape = (
                init_shape[0],
                init_shape[1] // div_factor,
                init_shape[2] // div_factor,
            )
            self.ops.append(
                DownsamplingTransfer(
                    shape=shape,
                    factor=factor,
                    dtype=dtype,
                    padding=padding,
                    device=device,
                )
            )
            div_factor *= factor

    def projection(self, x, n):
        return self.projection_between_levels(x, n_from=0, n_to=n)

    def prolongation(self, x, n):
        return self.projection_between_levels(x, n_from=n, n_to=0)

    def projection_n(self, x, n):
        return self.projection_between_levels(x, n_from=n, n_to=n + 1)

    def prolongation_n(self, x, n):
        return self.projection_between_levels(x, n_from=n + 1, n_to=n)

    def level_scale(self, n_from, n_to):
        """Integer scale factor between two pyramid levels."""
        return self.factor ** abs(n_to - n_from)

    def projection_between_levels(self, x, n_from, n_to):
        """
        Image-space transfer between pyramid levels.

        Fine -> coarse uses area interpolation. Coarse -> fine uses bilinear
        interpolation. This is appropriate for images such as x0/GT, but the
        coarse -> fine branch is not the adjoint of the fine -> coarse branch.
        """
        if n_from == n_to:
            return x

        scale = self.level_scale(n_from, n_to)

        if n_from < n_to:
            return F.interpolate(
                x,
                scale_factor=1 / scale,
                mode="area",
            )

        return F.interpolate(
            x,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

    def restriction_between_levels(self, x, n_from, n_to):
        """
        Linear restriction R from a finer/source level to a coarser/target level.

        This operator is intended for SSD scale-space formulas. It is separate
        from image-space transfer so that its adjoint can be defined exactly.
        """
        if n_from == n_to:
            return x
        if n_from > n_to:
            raise ValueError(
                "restriction_between_levels expects n_from <= n_to "
                f"(fine/source -> coarse/target), got {n_from} -> {n_to}."
            )

        scale = self.level_scale(n_from, n_to)
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def restriction_adjoint_between_levels(self, x, n_from, n_to):
        """
        Strict adjoint R^T of restriction_between_levels.

        If R averages each s x s block, then R^T repeats each coarse value back
        to an s x s block and divides by s^2, satisfying
        <R u, v> = <u, R^T v>.

        Args:
            x: Tensor on the coarse/source level.
            n_from: Coarse/source level of x.
            n_to: Finer/target level.
        """
        if n_from == n_to:
            return x
        if n_from < n_to:
            raise ValueError(
                "restriction_adjoint_between_levels expects n_from >= n_to "
                f"(coarse/source -> fine/target), got {n_from} -> {n_to}."
            )

        scale = self.level_scale(n_from, n_to)
        return F.interpolate(x, scale_factor=scale, mode="nearest") / (scale ** 2)

    def image_lift_between_levels(self, x, n_from, n_to):
        """Image-space lift/prolongation for clean images such as x0."""
        return self.projection_between_levels(x, n_from=n_from, n_to=n_to)


def _shape_smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 2, 1, 128, 128
    transfer = MultiLevelTransfer(n_level=3, init_shape=(C, H, W), device=device)

    x0 = torch.randn(B, C, H, W, device=device)

    x1 = transfer.projection_between_levels(x0, n_from=0, n_to=1)
    assert x1.shape == (B, C, H // 2, W // 2), x1.shape

    x2 = transfer.projection_between_levels(x1, n_from=1, n_to=2)
    assert x2.shape == (B, C, H // 4, W // 4), x2.shape

    x2_direct = transfer.projection_between_levels(x0, n_from=0, n_to=2)
    assert x2_direct.shape == (B, C, H // 4, W // 4), x2_direct.shape

    x1_back = transfer.projection_between_levels(x2, n_from=2, n_to=1)
    assert x1_back.shape == (B, C, H // 2, W // 2), x1_back.shape

    x0_back = transfer.projection_between_levels(x2, n_from=2, n_to=0)
    assert x0_back.shape == (B, C, H, W), x0_back.shape

    x0_same = transfer.projection_between_levels(x0, n_from=0, n_to=0)
    assert x0_same.shape == x0.shape
    assert torch.allclose(x0_same, x0)

    u = torch.randn(B, C, H, W, device=device)
    v = torch.randn(B, C, H // 2, W // 2, device=device)
    Ru = transfer.restriction_between_levels(u, n_from=0, n_to=1)
    RTv = transfer.restriction_adjoint_between_levels(v, n_from=1, n_to=0)
    lhs = (Ru * v).sum()
    rhs = (u * RTv).sum()
    relerr = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-12)
    assert relerr < 1e-5, relerr.item()

    print("MultiLevelTransfer shape smoke test passed.")


if __name__ == "__main__":
    _shape_smoke_test()