#!/usr/bin/env python3
"""Architecture-neutral NNX example using TE's public dot_general helper."""

from __future__ import annotations

import argparse
import os

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import te_quantization


class ExistingProjection(nnx.Module):
    """A custom projection that keeps ownership of its existing parameters."""

    def __init__(self, in_features: int, out_features: int, *, dot_general, rngs: nnx.Rngs):
        self.kernel = nnx.Param(
            jax.nn.initializers.lecun_normal()(rngs.params(), (in_features, out_features)),
        )
        self.dot_general = dot_general

    def __call__(self, inputs):
        return self.dot_general(
            inputs,
            self.kernel.value.astype(inputs.dtype),
            (((inputs.ndim - 1,), (0,)), ((), ())),
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
    projection = ExistingProjection(
        128,
        256,
        dot_general=dot_general,
        rngs=nnx.Rngs(0),
    )
    graphdef, state = nnx.split(projection)
    te_quantization.reset_stats()

    @jax.jit
    def apply(state, inputs):
        return nnx.merge(graphdef, state)(inputs)

    inputs = jax.random.normal(
        jax.random.key(1),
        (32, 128),
        dtype=jnp.bfloat16,
    )
    output = apply(state, inputs)
    jax.block_until_ready(output)
    print(
        f"recipe={te_quantization.selected_recipe_name()} "
        f"input={inputs.shape} output={output.shape} "
        f"sites={te_quantization.GEMM_STATS}"
    )


if __name__ == "__main__":
    main()
