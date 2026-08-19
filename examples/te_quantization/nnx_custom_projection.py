#!/usr/bin/env python3
"""Architecture-neutral NNX example using TE's public dot_general helper."""

from __future__ import annotations

import argparse
import os

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import te_quantization


class ExistingMultiAxisProjection(nnx.Module):
    """A custom projection with an existing higher-rank kernel."""

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        contraction_axes: int | tuple[int, ...],
        *,
        dot_general,
        rngs: nnx.Rngs,
    ):
        self.kernel = nnx.Param(
            jax.nn.initializers.lecun_normal()(rngs.params(), kernel_shape),
        )
        self.contraction_axes = (contraction_axes,) if isinstance(contraction_axes, int) else contraction_axes
        self.dot_general = dot_general

    def __call__(self, inputs):
        lhs_contract = tuple(range(inputs.ndim - len(self.contraction_axes), inputs.ndim))
        return self.dot_general(
            inputs,
            self.kernel.value.astype(inputs.dtype),
            ((lhs_contract, self.contraction_axes), ((), ())),
        )


class ExampleModel(nnx.Module):
    """Exercise a native Linear and two custom multi-axis projections."""

    def __init__(self, *, dot_general, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            128,
            256,
            use_bias=False,
            dtype=jnp.bfloat16,
            dot_general=dot_general,
            rngs=rngs,
        )
        # [batch, 48] x [48, 3, 16] -> [batch, 3, 16]
        self.expand = ExistingMultiAxisProjection(
            (48, 3, 16),
            0,
            dot_general=dot_general,
            rngs=rngs,
        )
        # [batch, 3, 16] x [3, 16, 80] -> [batch, 80]
        self.collapse = ExistingMultiAxisProjection(
            (3, 16, 80),
            (0, 1),
            dot_general=dot_general,
            rngs=rngs,
        )

    def __call__(self, linear_inputs, multi_axis_inputs):
        return (
            self.linear(linear_inputs),
            self.collapse(self.expand(multi_axis_inputs)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("current", "mxfp8", "nvfp4"),
        default="mxfp8",
    )
    args = parser.parse_args()
    os.environ["OPENPI_TE_RECIPE"] = args.recipe

    dot_general = te_quantization.make_nnx_dot_general()
    model = ExampleModel(
        dot_general=dot_general,
        rngs=nnx.Rngs(0),
    )
    graphdef, state = nnx.split(model)
    te_quantization.reset_stats()

    @jax.jit
    def apply(state, linear_inputs, multi_axis_inputs):
        return nnx.merge(graphdef, state)(linear_inputs, multi_axis_inputs)

    linear_inputs = jax.random.normal(
        jax.random.key(1),
        (32, 128),
        dtype=jnp.bfloat16,
    )
    multi_axis_inputs = jax.random.normal(
        jax.random.key(2),
        (32, 48),
        dtype=jnp.bfloat16,
    )
    linear_output, multi_axis_output = apply(
        state,
        linear_inputs,
        multi_axis_inputs,
    )
    jax.block_until_ready((linear_output, multi_axis_output))
    print(
        f"recipe={te_quantization.selected_recipe_name()} "
        f"linear={linear_inputs.shape}->{linear_output.shape} "
        f"multi_axis={multi_axis_inputs.shape}->{multi_axis_output.shape} "
        f"sites={te_quantization.GEMM_STATS}"
    )


if __name__ == "__main__":
    main()
